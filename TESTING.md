# Testing Documentation

## Overview

This project has comprehensive test coverage using pytest. The test suite achieves **93% code coverage** with 46 tests covering all major functionality.

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
pytest
```

### Run Tests with Coverage Report

```bash
pytest --cov=censor --cov-report=term-missing
```

### Run Tests with HTML Coverage Report

```bash
pytest --cov=censor --cov-report=html
# View the report at htmlcov/index.html
```

## Test Structure

The test suite is organized into the following test classes:

### Core Functionality Tests

- **TestCensorBotInit**: Tests initialization and hardware acceleration setup
  - CPU-only mode
  - CUDA GPU acceleration
  - Apple Silicon (CoreML) acceleration
  - Fallback when Whisper is unavailable

- **TestCensorBotBeepFile**: Tests beep sound file creation
  - Creation of beep file
  - Skipping when file already exists

### Video Processing Tests

- **TestExtractEmbeddedSubtitles**: Tests subtitle extraction from video files
  - Successful extraction
  - Handling missing subtitles
  - Error handling

- **TestDownloadSubtitles**: Tests online subtitle downloading
  - Successful download from OpenSubtitles
  - Handling unavailable subtitles
  - Network error handling

- **TestExtractAudio**: Tests audio extraction from video
  - Successful extraction
  - Failure handling

### Content Analysis Tests

- **TestLoadBadwords**: Tests bad word list loading
  - Default word list
  - Custom word list from file
  - File not found handling

- **TestProcessSubtitles**: Tests subtitle processing for profanity
  - Detection of profanity in subtitles
  - Clean subtitles
  - Error handling

- **TestTranscribeAudio**: Tests Whisper transcription
  - Successful transcription with timestamps
  - Handling missing Whisper installation

- **TestFindCensorSegments**: Tests censor segment identification
  - Finding profanity in transcription
  - Handling clean content
  - Padding configuration
  - Boundary conditions

### Output Generation Tests

- **TestApplyCensorship**: Tests censorship application
  - Mute mode
  - Beep mode
  - Dual audio track creation
  - Error handling
  - Empty segment handling

### Integration Tests

- **TestProcessVideo**: Tests end-to-end video processing
  - Processing with provided subtitles
  - Processing with Whisper transcription
  - Error handling

- **TestMain**: Tests CLI interface
  - Single file processing
  - Various command-line options
  - Batch processing
  - Error conditions

### Unit Tests

- **TestDefaultBadwords**: Tests the default profanity word list
- **TestModuleConstants**: Tests module-level constants and feature flags

## Coverage Report

Current coverage: **93%**

### Covered Areas

- All core functionality (100%)
- Video and audio processing (100%)
- Subtitle handling (100%)
- Profanity detection (100%)
- CLI argument parsing (100%)
- Error handling (100%)

### Uncovered Lines

The remaining 7% consists of:
- Import error handlers for optional dependencies (coremltools)
- Some edge cases in conditional branches
- Platform-specific code paths that cannot be tested in the current environment

## Test Mocking Strategy

The test suite uses extensive mocking to:
- Avoid dependency on external services (OpenSubtitles)
- Avoid dependency on system tools (ffmpeg)
- Avoid dependency on heavy ML models (Whisper)
- Test error conditions safely

Mock libraries used:
- `unittest.mock` for function and method mocking
- `pytest-mock` for pytest-specific mocking features

## Continuous Integration

To run tests in CI/CD:

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Run tests with coverage
pytest --cov=censor --cov-report=xml --cov-report=term

# Coverage threshold check (optional)
pytest --cov=censor --cov-fail-under=90
```

## Adding New Tests

When adding new functionality:

1. Add tests in the appropriate test class
2. Use the `mock_censorbot` fixture for tests requiring a CensorBot instance
3. Mock external dependencies (subprocess, file I/O, network calls)
4. Verify coverage remains above 90%:
   ```bash
   pytest --cov=censor --cov-report=term-missing
   ```

## Test Best Practices

1. **Use descriptive test names**: Test names should clearly describe what is being tested
2. **One assertion per test**: Each test should verify one specific behavior
3. **Mock external dependencies**: Never rely on external services or files in tests
4. **Test error conditions**: Always test both success and failure paths
5. **Use fixtures**: Reuse common setup code with pytest fixtures
