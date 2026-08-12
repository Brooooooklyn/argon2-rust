# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0](https://github.com/Brooooooklyn/argon2-rust/compare/v0.0.3...v1.0.0) - 2026-08-12

### Added

- SemVer policy for the `1.x` series (README): default-feature public API is
  stable; `internal-api` / `__internal` remain exempt; MSRV may rise on minor
- `documentation` URL and `[package.metadata.docs.rs]` so docs.rs builds with
  all features

### Changed

- **Stabilized** the public API at `1.0.0`. Breaking changes from here need a
  major bump. The surface is the typed `Params` builder (from 0.0.3), raw and
  PHC hash/verify entry points, bounded verify, pooled `Hasher`, OWASP and
  RFC 9106 presets, and C error-code parity on `Error`
- SECURITY.md: support policy is latest `1.x`, not pre-1.0 “latest only”

### Fixed

- README quick-start dependency pin (was still `0.0.2`)

## [0.0.3](https://github.com/Brooooooklyn/argon2-rust/compare/v0.0.2...v0.0.3) - 2026-08-11

### Added

- *(params)* [**breaking**] replace the positional constructors with a typed builder ([#10](https://github.com/Brooooooklyn/argon2-rust/pull/10))

### Other

- replace tag-based publish with release-plz ([#11](https://github.com/Brooooooklyn/argon2-rust/pull/11))
- plan the Params API redesign
- spec the Params API redesign
