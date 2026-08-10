//! Portable scalar-vs-SIMD Base64 shootout.
//!
//! This deliberately has no Criterion dependency so the exact same optimized
//! workload runs natively, under Wasmtime, and in a minimal Linux container:
//!
//! ```text
//! cargo bench --bench base64_shootout
//! CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime \
//!   RUSTFLAGS='-C target-feature=+simd128' \
//!   cargo bench --target wasm32-wasip1 --bench base64_shootout
//! ```
//!
//! `auto` is the production entry point and therefore includes the cached
//! dispatch load. Named backends bypass selection so multiple x86 kernels can
//! be compared in one process. Every backend is checked against scalar before
//! it is timed.

use std::hint::black_box;
use std::time::{Duration, Instant};

use argon2_rust::__internal::{
    Base64Backend, b64_len, base64_backend, from_base64, from_base64_with_backend, to_base64,
    to_base64_with_backend,
};

const LENGTHS: [usize; 5] = [8, 16, 32, 256, 4096];
const SAMPLES: usize = 9;
const TARGET_SAMPLE: Duration = Duration::from_millis(20);

#[derive(Copy, Clone)]
enum Operation {
    Encode,
    Decode,
}

impl Operation {
    const ALL: [Self; 2] = [Self::Encode, Self::Decode];

    const fn name(self) -> &'static str {
        match self {
            Self::Encode => "encode",
            Self::Decode => "decode",
        }
    }
}

#[derive(Copy, Clone)]
enum Mode {
    Scalar,
    Auto,
    Backend(Base64Backend),
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Auto => "auto",
            Self::Backend(backend) => backend.name(),
        }
    }
}

struct Case {
    raw: Vec<u8>,
    encoded: Vec<u8>,
    encode_dst: Vec<u8>,
    decode_dst: Vec<u8>,
}

impl Case {
    fn new(len: usize) -> Self {
        let raw: Vec<_> = (0..len)
            .map(|i| (i as u8).wrapping_mul(37) ^ 0x5a)
            .collect();
        let mut encoded = vec![0u8; b64_len(len as u32) + 1];
        let written = to_base64(&mut encoded, &raw).expect("sized encoding destination");
        encoded.truncate(written);

        Self {
            raw,
            encode_dst: vec![0; written + 1],
            decode_dst: vec![0; len],
            encoded,
        }
    }

    fn run(&mut self, operation: Operation, mode: Mode, iterations: u64) -> u8 {
        let mut checksum = 0u8;
        for _ in 0..iterations {
            match (operation, mode) {
                (Operation::Encode, Mode::Scalar) => {
                    // SAFETY: scalar has no CPU feature precondition.
                    let n = unsafe {
                        to_base64_with_backend(
                            black_box(&mut self.encode_dst),
                            black_box(&self.raw),
                            Base64Backend::Scalar,
                        )
                    }
                    .expect("encoding succeeds");
                    checksum ^= self.encode_dst[black_box(n - usize::from(n != 0))];
                }
                (Operation::Encode, Mode::Auto) => {
                    let n = to_base64(black_box(&mut self.encode_dst), black_box(&self.raw))
                        .expect("encoding succeeds");
                    checksum ^= self.encode_dst[black_box(n - usize::from(n != 0))];
                }
                (Operation::Encode, Mode::Backend(backend)) => {
                    // SAFETY: main only constructs this mode after checking
                    // `backend.is_available()` in this process.
                    let n = unsafe {
                        to_base64_with_backend(
                            black_box(&mut self.encode_dst),
                            black_box(&self.raw),
                            backend,
                        )
                    }
                    .expect("encoding succeeds");
                    checksum ^= self.encode_dst[black_box(n - usize::from(n != 0))];
                }
                (Operation::Decode, Mode::Scalar) => {
                    // SAFETY: scalar has no CPU feature precondition.
                    let (n, consumed) = unsafe {
                        from_base64_with_backend(
                            black_box(&mut self.decode_dst),
                            black_box(&self.encoded),
                            Base64Backend::Scalar,
                        )
                    }
                    .expect("decoding succeeds");
                    checksum ^=
                        self.decode_dst[black_box(n - usize::from(n != 0))] ^ (consumed as u8);
                }
                (Operation::Decode, Mode::Auto) => {
                    let (n, consumed) =
                        from_base64(black_box(&mut self.decode_dst), black_box(&self.encoded))
                            .expect("decoding succeeds");
                    checksum ^=
                        self.decode_dst[black_box(n - usize::from(n != 0))] ^ (consumed as u8);
                }
                (Operation::Decode, Mode::Backend(backend)) => {
                    // SAFETY: as in the explicit encode arm above.
                    let (n, consumed) = unsafe {
                        from_base64_with_backend(
                            black_box(&mut self.decode_dst),
                            black_box(&self.encoded),
                            backend,
                        )
                    }
                    .expect("decoding succeeds");
                    checksum ^=
                        self.decode_dst[black_box(n - usize::from(n != 0))] ^ (consumed as u8);
                }
            }
        }
        black_box(checksum)
    }
}

fn elapsed(case: &mut Case, operation: Operation, mode: Mode, iterations: u64) -> Duration {
    let started = Instant::now();
    black_box(case.run(operation, mode, iterations));
    started.elapsed()
}

fn iterations_for(case: &mut Case, operation: Operation) -> u64 {
    let probe_iterations = 256;
    let probe = elapsed(case, operation, Mode::Scalar, probe_iterations);
    let nanos_per_iteration = (probe.as_nanos() / u128::from(probe_iterations)).max(1);
    let iterations = TARGET_SAMPLE.as_nanos() / nanos_per_iteration;
    u64::try_from(iterations.clamp(256, 20_000_000)).expect("bounded iteration count")
}

fn median(mut values: [f64; SAMPLES]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    values[SAMPLES / 2]
}

fn validate(case: &mut Case, backend: Base64Backend) {
    let mut scalar_encoded = vec![0; case.encode_dst.len()];
    let mut backend_encoded = vec![0; case.encode_dst.len()];
    // SAFETY: scalar has no CPU feature precondition; the caller checked the
    // named backend before reaching this function.
    let scalar_len =
        unsafe { to_base64_with_backend(&mut scalar_encoded, &case.raw, Base64Backend::Scalar) }
            .expect("scalar encoding succeeds");
    // SAFETY: the caller checked this backend with `is_available()`.
    let backend_len = unsafe { to_base64_with_backend(&mut backend_encoded, &case.raw, backend) }
        .expect("backend encoding succeeds");
    assert_eq!(backend_len, scalar_len);
    assert_eq!(backend_encoded, scalar_encoded);

    let mut scalar_decoded = vec![0; case.decode_dst.len()];
    let mut backend_decoded = vec![0; case.decode_dst.len()];
    // SAFETY: the same availability proof applies to both calls.
    let scalar_result = unsafe {
        from_base64_with_backend(&mut scalar_decoded, &case.encoded, Base64Backend::Scalar)
    }
    .expect("scalar decoding succeeds");
    // SAFETY: the caller checked this backend with `is_available()`.
    let backend_result =
        unsafe { from_base64_with_backend(&mut backend_decoded, &case.encoded, backend) }
            .expect("backend decoding succeeds");
    assert_eq!(backend_result, scalar_result);
    assert_eq!(backend_decoded, scalar_decoded);
}

fn measure(case: &mut Case, operation: Operation, mode: Mode, iterations: u64) -> (f64, f64) {
    // Warm both paths and resolve the production backend before sampling.
    black_box(case.run(operation, Mode::Scalar, 64));
    black_box(case.run(operation, mode, 64));

    let mut scalar = [0.0; SAMPLES];
    let mut candidate = [0.0; SAMPLES];
    for sample in 0..SAMPLES {
        let (scalar_elapsed, candidate_elapsed) = if sample % 2 == 0 {
            (
                elapsed(case, operation, Mode::Scalar, iterations),
                elapsed(case, operation, mode, iterations),
            )
        } else {
            let candidate_elapsed = elapsed(case, operation, mode, iterations);
            let scalar_elapsed = elapsed(case, operation, Mode::Scalar, iterations);
            (scalar_elapsed, candidate_elapsed)
        };
        scalar[sample] = scalar_elapsed.as_secs_f64() * 1e9 / iterations as f64;
        candidate[sample] = candidate_elapsed.as_secs_f64() * 1e9 / iterations as f64;
    }
    (median(scalar), median(candidate))
}

fn main() {
    let selected = base64_backend();
    println!(
        "base64_shootout arch={} selected={} samples={} target_ms={}",
        std::env::consts::ARCH,
        selected,
        SAMPLES,
        TARGET_SAMPLE.as_millis()
    );
    println!("operation raw_len mode scalar_ns candidate_ns speedup");

    for len in LENGTHS {
        let mut case = Case::new(len);
        for &backend in Base64Backend::ALL {
            if backend.is_available() {
                validate(&mut case, backend);
            }
        }

        for operation in Operation::ALL {
            let iterations = iterations_for(&mut case, operation);
            let (scalar, candidate) = measure(&mut case, operation, Mode::Auto, iterations);
            println!(
                "{} {} {} {:.3} {:.3} {:.3}",
                operation.name(),
                len,
                Mode::Auto.name(),
                scalar,
                candidate,
                scalar / candidate
            );

            for &backend in Base64Backend::ALL {
                if backend == Base64Backend::Scalar || !backend.is_available() {
                    continue;
                }
                let mode = Mode::Backend(backend);
                let (scalar, candidate) = measure(&mut case, operation, mode, iterations);
                println!(
                    "{} {} {} {:.3} {:.3} {:.3}",
                    operation.name(),
                    len,
                    mode.name(),
                    scalar,
                    candidate,
                    scalar / candidate
                );
            }
        }
    }
}
