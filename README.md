# CensorBot

A powerful Python-based tool for automatically censoring profanity in video files. CensorBot uses a multi-stage approach to detect and censor inappropriate language, prioritizing accuracy and performance.

## Features

### Core Functionality
- **Multiple Detection Methods**:
  1. Embedded Subtitles: Extracts and uses subtitles from the video file
  2. Online Subtitles: Downloads matching subtitles from OpenSubtitles
  3. AI Transcription: Falls back to Whisper-based transcription if no subtitles are found

- **Flexible Censoring Options**:
  - Mute Mode: Silences the offensive segments
  - Beep Mode: Replaces offensive words with a beep sound
  - Dual Audio: Keeps both original and censored audio tracks

### Input/Output Support
- **Video Formats**: MP4, MKV, AVI
- **Subtitle Formats**: SRT (embedded or external)
- **Output**: Maintains original video quality with censored audio

### Performance Features
- Cross-platform hardware acceleration:
  - NVIDIA GPUs: CUDA acceleration
  - Apple Silicon: CoreML and Neural Engine
  - Intel/AMD: Multi-threaded CPU processing
- Parallel processing for batch operations
- Progress tracking for long operations
- Efficient memory management

## Prerequisites

### Required
- Docker
- 4GB RAM minimum
- 10GB free disk space

### Optional (Platform Specific)
- NVIDIA GPU with CUDA support
- Apple Silicon (M1/M2) Mac
- NVIDIA Container Toolkit (for NVIDIA GPUs)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/censorbot.git
cd censorbot
```

2. Build the Docker image for your platform:

For AMD64 (Intel/AMD) or ARM64 (Apple Silicon):
```bash
docker buildx build --platform $(uname -m) -t censorbot .
```

For multi-platform build:
```bash
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 -t censorbot --push .
```

## Usage

### Quick Start
Process a single video with default settings (auto-detects best acceleration):
```bash
docker run -v $(pwd):/app censorbot -i input.mp4
```

### Platform-Specific Usage

1. **NVIDIA GPU (Linux/Windows)**
```bash
docker run --gpus all -v $(pwd):/app censorbot -i input.mp4 --gpu
```

2. **Apple Silicon (M1/M2 Mac)**
```bash
# Automatically uses CoreML acceleration
docker run -v $(pwd):/app censorbot -i input.mp4
```

3. **Force CPU Processing (Any Platform)**
```bash
docker run -v $(pwd):/app censorbot -i input.mp4 --force-cpu
```

### Common Use Cases

1. **Using Beep Sound Instead of Muting**
```bash
docker run -v $(pwd):/app censorbot -i input.mp4 --mode beep
```

2. **Process All Videos in a Directory**
```bash
docker run -v $(pwd):/app censorbot -i /app/videos --batch --max-workers 4
```

3. **Using Custom Subtitle File**
```bash
docker run -v $(pwd):/app censorbot -i input.mp4 -s subtitles.srt
```

### Advanced Options

#### Command Line Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `-i, --input` | Input video file or directory | Required |
| `-o, --output` | Output path | input_censored.mp4 |
| `-w, --wordlist` | Additional words to censor | Built-in list |
| `-s, --subtitles` | External subtitle file | None |
| `--gpu` | Enable GPU acceleration | Auto-detect |
| `--force-cpu` | Disable hardware acceleration | False |
| `--model` | Whisper model size (tiny/base/small/medium/large) | base |
| `--padding` | Padding around censored segments (seconds) | 0.2 |
| `--no-download` | Disable subtitle downloading | False |
| `--mode` | Censoring mode (mute/beep) | mute |
| `--single-audio` | Only keep censored audio track | False |
| `--batch` | Process all videos in directory | False |
| `--max-workers` | Number of parallel workers | 1 |

## Performance Optimization

### Platform-Specific Acceleration

#### NVIDIA GPUs
- Ensure NVIDIA drivers and Container Toolkit are installed
- Use `--gpu` flag for CUDA acceleration
- Adjust model size based on available VRAM

#### Apple Silicon (M1/M2)
- Automatically uses CoreML and Neural Engine
- No additional flags needed
- Optimized for efficiency and battery life

#### CPU-Only Systems
- Uses multi-threading automatically
- Adjust `--max-workers` based on CPU cores
- Consider using smaller model sizes

### Batch Processing
For multiple videos:
1. Use the `--batch` flag
2. Set `--max-workers` based on:
   - CPU cores available
   - GPU memory (if using CUDA)
   - System memory

## Troubleshooting

### Platform-Specific Issues

1. **NVIDIA GPU Issues**
   - Verify NVIDIA drivers are installed
   - Check NVIDIA Container Toolkit installation
   - Ensure `--gpus all` flag is used
   - Monitor GPU memory usage

2. **Apple Silicon Issues**
   - Ensure using ARM64 version of Docker
   - Check CoreML installation
   - Monitor system temperature
   - Consider `--force-cpu` if experiencing issues

3. **General Performance Issues**
   - Check hardware acceleration is working
   - Monitor system resources
   - Adjust model size and workers
   - Consider platform-specific optimizations

### Common Error Messages

- `Failed to extract audio`: Check video file format and permissions
- `No embedded subtitles found`: Video doesn't contain subtitles
- `Failed to download subtitles`: Network or compatibility issue
- `CUDA out of memory`: Reduce model size or batch size
- `CoreML error`: Check macOS version and permissions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI's Whisper for transcription
- FFmpeg for audio processing
- OpenSubtitles for subtitle database
- Subliminal for subtitle downloading
- Apple's CoreML for M1/M2 acceleration
- NVIDIA CUDA for GPU acceleration 