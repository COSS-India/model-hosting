# Changelog

All notable changes to the MLflow ASR Service will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Docker deployment support
- Comprehensive documentation (README, SETUP, USAGE, DOCKER guides)
- Test scripts for API testing
- HuggingFace token support via environment variables
- Audio loading using Python's wave module (removes FFmpeg dependency)
- Health check endpoint
- Docker Compose configuration

### Changed
- Updated audio loading to use `wave` module instead of `torchaudio.load()` to avoid torchcodec dependency
- Improved error handling in model wrapper
- Enhanced documentation structure

### Fixed
- Fixed missing `onnxruntime` dependency in Docker image
- Fixed audio loading issues with torchcodec dependency
- Fixed HuggingFace token handling in Docker containers

## [1.0.0] - 2024-12-08

### Added
- Initial release of MLflow ASR Service
- Support for `ai4bharat/indic-conformer-600m-multilingual` model
- MLflow PyFunc model wrapper implementation
- REST API endpoint for audio transcription
- Support for multiple Indic languages
- CTC and greedy decoding strategies
- GPU support (automatic detection)
- Model logging script
- Basic test script

### Features
- Multilingual ASR (Hindi, Tamil, Telugu, etc.)
- Flexible decoding (CTC, greedy)
- GPU acceleration
- RESTful API
- MLflow integration

---

## Version History

### v1.0.0 (2024-12-08)
- Initial release
- Basic ASR functionality
- MLflow integration
- Docker support

---

## Notes

- All dates are in YYYY-MM-DD format
- Breaking changes are marked with ⚠️
- Security fixes are marked with 🔒

---

**Last Updated**: December 2024






