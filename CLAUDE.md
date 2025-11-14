# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CensorBot is a Python-based video profanity censoring tool that uses a multi-stage detection approach:
1. **Embedded subtitles** (extracted from video)
2. **Downloaded subtitles** (from OpenSubtitles via Subliminal)
3. **AI transcription** (Whisper-based fallback with hardware acceleration)

**Hardware Acceleration Support**:
- **Apple Silicon**: MLX framework with Metal/Neural Engine acceleration
- **NVIDIA GPUs**: CUDA acceleration with faster-whisper
- **CPU**: Multi-threaded processing fallback

## Development Commands

### Docker Build
```bash
# Build for current platform
docker buildx build --platform $(uname -m) -t censorbot .

# Multi-platform build
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 -t censorbot --push .
```

### Running the Tool
```bash
# Basic usage (auto-detects acceleration)
docker run -v $(pwd):/app censorbot -i input.mp4 -o output.mp4

# NVIDIA GPU acceleration
docker run --gpus all -v $(pwd):/app censorbot -i input.mp4 -o output.mp4

# Force CPU processing
docker run -v $(pwd):/app censorbot -i input.mp4 -o output.mp4 --force-cpu

# Beep mode instead of mute
docker run -v $(pwd):/app censorbot -i input.mp4 -o output.mp4 --mode beep

# Use custom wordlist
docker run -v $(pwd):/app censorbot -i input.mp4 -o output.mp4 -w badwords.txt

# Use external subtitles
docker run -v $(pwd):/app censorbot -i input.mp4 -o output.mp4 -s subtitles.srt

# Single audio track (censored only)
docker run -v $(pwd):/app censorbot -i input.mp4 -o output.mp4 --single-audio
```

### Testing Changes
```bash
# Run directly with Python (for development)
python3 censor.py -i test_video.mp4 -o output.mp4

# Test with custom wordlist
python3 censor.py -i test_video.mp4 -o output.mp4 -w custom_badwords.txt

# Test with external subtitles
python3 censor.py -i test_video.mp4 -o output.mp4 -s subtitles.srt

# Test beep mode
python3 censor.py -i test_video.mp4 -o output.mp4 --mode beep

# Force CPU mode (for testing without GPU)
python3 censor.py -i test_video.mp4 -o output.mp4 --force-cpu
```

## Architecture

### Core Processing Pipeline
The `CensorBot` class implements a three-stage subtitle acquisition strategy:

1. **extract_embedded_subtitles()** - Uses FFmpeg to extract subtitle streams from video
2. **download_subtitles()** - Falls back to Subliminal + OpenSubtitles API
3. **transcribe_audio()** - Final fallback using Faster-Whisper with word-level timestamps

### Hardware Acceleration Strategy (REWRITTEN)
Platform detection and acceleration setup happens in `setup_acceleration()`:

**Apple Silicon (MLX Backend)**:
- Uses `mlx-whisper` with native Metal/Neural Engine acceleration
- TRUE hardware acceleration (not fake "CoreML" claims)
- Significantly faster than CPU-only processing
- Automatically detected on ARM64 Darwin systems

**NVIDIA GPUs (CUDA Backend)**:
- Uses `faster-whisper` with CUDA acceleration
- Float16 compute type for optimal performance
- Requires `--gpus all` flag in Docker

**CPU Fallback**:
- Uses `faster-whisper` with multi-threaded CPU processing
- INT8 quantization for reduced memory usage
- Automatically uses all available CPU cores

### Audio Processing Modes (SIMPLIFIED & FIXED)
Two censoring approaches using simplified FFmpeg filter chains:

**Mute Mode** (default):
- Chains volume filters: `volume=enable='between(t,start,end)':volume=0,...`
- Clean, accurate silencing of profanity
- Single FFmpeg command, no batching complexity

**Beep Mode**:
- Creates continuous sine wave, enables at censor points
- Mixes beep with muted audio using filter_complex
- 1000Hz tone at 30% volume
- Accurate timing without complex adelay calculations

**Output Format**:
- Dual audio tracks by default (original + censored)
- Use `--single-audio` to keep only censored track
- Video stream copied (no re-encoding)

## Key Implementation Details

### Dockerfile Multi-Stage Build (IMPROVED)
The Dockerfile uses platform-specific build stages with runtime detection:

**Build Stages**:
- `base`: Common Ubuntu 22.04 foundation with FFmpeg
- `apple_silicon`: ARM64 with CPU PyTorch + MLX for Metal acceleration
- `nvidia`: CUDA 12.1 base with GPU-enabled PyTorch
- `cpu_only`: AMD64 with CPU-only PyTorch
- `final`: Lightweight runtime with conditional package installation

**Key Improvements**:
- Runtime platform detection (not build-time copying)
- Smaller final image size
- Automatic MLX installation on ARM64
- Conditional CUDA support on AMD64

### Subtitle Caching
Subliminal uses a DBM cache (`cachefile.dbm`) to avoid re-downloading subtitles. The cache directory is created at `/root/.cache/subliminal` in the Docker container.

### Word Matching
The default badwords list (`DEFAULT_BADWORDS` in censor.py) includes:
- Common profanity and variations (f*ck, sh*t)
- Racial slurs
- Sexual terms
- Compound words (motherfucker, bullshit)

Additional words can be loaded from `badwords.txt` or custom file via `-w` flag.

### Timestamp Alignment
Default padding is 0.2s before/after detected segments to account for:
- Whisper timestamp variance
- Subtitle sync drift
- Audio attack/decay times

Configurable via `--padding` argument.

## Dependencies (UPDATED)

### Critical Version Constraints
- **numpy**: Pinned to `<2.0.0` for faster-whisper compatibility
- **torch**: `>=2.2.0` with platform-specific index URLs
- **faster-whisper**: `>=1.0.0` (updated from 0.9.0)
- **mlx-whisper**: `>=0.4.0` for Apple Silicon only

### Platform-Specific Packages

**Apple Silicon (Darwin ARM64)**:
- `mlx>=0.4.0`: Apple's ML framework for Metal acceleration
- `mlx-whisper>=0.4.0`: Whisper with native Metal support
- `torch>=2.2.0` (CPU index): No CUDA overhead

**NVIDIA GPUs (Linux AMD64)**:
- `torch>=2.2.0` (CUDA 12.1 index): GPU-accelerated PyTorch
- CUDA runtime from base image

**CPU-Only (All Platforms)**:
- `torch>=2.2.0` (CPU index): Lightweight PyTorch
- `faster-whisper>=1.0.0`: Multi-threaded transcription

## Working with Video Formats

### Supported Input Formats
- MP4, MKV, AVI (any format FFmpeg can decode)
- Must contain at least one audio stream
- Subtitles can be embedded or external (SRT format)

### Output Format
Always outputs MP4 with:
- Original video stream (copied, not re-encoded)
- Dual audio tracks (stream 0: original, stream 1: censored)
- Use `--single-audio` to keep only censored track

## Code Architecture

### Main Function Flow
The rewritten implementation follows a clean, linear flow:

1. **Initialization**: `CensorBot.__init__()` sets up hardware acceleration
2. **Subtitle Acquisition**: 3-stage fallback (embedded → download → transcribe)
3. **Audio Extraction**: Extract audio from video to WAV
4. **Segment Detection**: Find profanity timestamps from subtitles or transcription
5. **Audio Censoring**: Apply mute/beep using simplified FFmpeg filters
6. **Video Merging**: Merge censored audio back with video
7. **Subtitle Embedding**: Optionally embed censored subtitles

### Critical Implementation Notes

**What Was Fixed**:
- ❌ Removed broken batching logic (382 lines → 70 lines)
- ❌ Removed fake "CoreML acceleration" claims
- ❌ Fixed audio output format (single censored track, not dual-channel merge)
- ❌ Removed process_video() function (unified execution path)
- ✅ Simplified FFmpeg filter chains (chained volume filters)
- ✅ Added real MLX acceleration for Apple Silicon
- ✅ Clean error handling and logging

**FFmpeg Filter Strategy**:
- **Old**: Batch segments → create temp files → concatenate (broken)
- **New**: Chain volume filters in single command (works correctly)

Example mute filter:
```bash
-af "volume=enable='between(t,5,10)':volume=0,volume=enable='between(t,15,20)':volume=0"
```

Example beep filter:
```bash
-filter_complex "sine=frequency=1000:duration=300[sine];[sine]volume=enable='between(t,5,10)':volume=0.3[beep];[0:a]volume=enable='between(t,5,10)':volume=0[muted];[muted][beep]amix=inputs=2[out]"
```

## Development Guidelines

### Testing Censorship Logic
```bash
# Test with known profanity
echo "This is a fucking test" > test.srt
# Convert to proper SRT format and test

# Verify FFmpeg filter output
ffmpeg -i input.wav -af "volume=enable='between(t,5,10)':volume=0" output.wav
ffprobe output.wav  # Check audio is valid
```

### Adding New Bad Words
Edit `badwords.txt` or pass custom file via `-w`:
```bash
python3 censor.py -i video.mp4 -o output.mp4 -w custom_words.txt
```

### Debugging Hardware Acceleration
Check logs for acceleration backend:
```
✓ Using Apple Silicon Metal acceleration (MLX)
✓ Using NVIDIA CUDA acceleration
Using CPU multi-threaded processing
```

If MLX/CUDA not detected when expected:
1. Verify platform: `uname -m` (arm64/aarch64 for Apple Silicon)
2. Check CUDA: `torch.cuda.is_available()` in Python
3. Check MLX: `python3 -c "import mlx_whisper"` (should not error)

## Recent Changes (2024 Rewrite)

Complete rewrite addressing critical bugs:
- **censor.py**: Full rewrite with simplified logic and real acceleration
- **Dockerfile**: Improved multi-stage build with runtime detection
- **requirements.txt**: Updated to modern versions with MLX support

All changes focused on fixing broken censoring functionality and adding genuine hardware acceleration.
