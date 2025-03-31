# Changelog

All notable changes to the Docling Actor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-03-09

### Changed

- Switched from full Docling CLI to docling-serve API
- Using the official quay.io/ds4sd/docling-serve-cpu Docker image
- Reduced Docker image size (from ~6GB to ~4GB)
- Implemented multi-stage Docker build to handle dependencies
- Improved Docker build process to ensure compatibility with docling-serve-cpu image
- Added new Python processor script for reliable API communication and content extraction
- Enhanced response handling with better content extraction logic
- Fixed ES modules compatibility issue with Apify CLI
- Added explicit tmpfs volume for temporary files
- Fixed environment variables format in actor.json
- Created optimized dependency installation approach
- Improved API compatibility with docling-serve
  - Updated endpoint from custom `/convert` to standard `/v1alpha/convert/source`
  - Revised JSON payload structure to match docling-serve API format
  - Added proper output field parsing based on format
- Enhanced startup process with health checks
- Added configurable API host and port through environment variables
- Better content type handling for different output formats
- Updated error handling to align with API responses

### Fixed

- Fixed actor input file conflict in get_actor_input(): now checks for and removes an existing /tmp/actor-input/INPUT directory if found, ensuring valid JSON input parsing.

### Technical Details

- Actor Specification v1
- Using quay.io/ds4sd/docling-serve-cpu:latest base image
- Node.js 20.x for Apify CLI
- Eliminated Python dependencies
- Simplified Docker build process

## [1.0.0] - 2025-02-07

### Added

- Initial release of Docling Actor
- Support for multiple document formats (PDF, DOCX, images)
- OCR capabilities for scanned documents
- Multiple output formats (md, json, html, text, doctags)
- Comprehensive error handling and logging
- Dataset records with processing status
- Memory monitoring and resource optimization
- Security features including non-root user execution

### Technical Details

- Actor Specification v1
- Docling v2.17.0
- Python 3.11
- Node.js 20.x
- Comprehensive error codes:
  - 10: Invalid input
  - 11: URL inaccessible
  - 12: Docling processing failed
  - 13: Output file missing
  - 14: Storage operation failed
  - 15: OCR processing failed
