# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Fixed
- Removed CMake compile definitions that caused issues on the `aarch64` architecture.
- Changed the download link to the ONNX Runtime in the README.md for the `aarch64` architecture.

### Added
- Added the `yolo-cls` project (pre 1.0.0 version)
- Added changelog (CHANGELOG.md)
- Added `Platform support` section in the README.md
- Added Windows (`x86_64`, CPU, GCC, MinGW) support.

### Changed
- Changed the vscode project settings.
- Changed the behavior of the `yolo` constructor (`src/yolo.cpp`, `yolo::yolo`).
  Now the constructor loads the ONNX model into the memory.
