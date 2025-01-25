# CensorBot

A powerful Python-based tool for automatically censoring profanity in video files. CensorBot uses a multi-stage approach to detect and censor inappropriate language, prioritizing accuracy and performance.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-donate-yellow.svg)](https://www.buymeacoffee.com/yourusername)

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

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Support the Project

If you find this tool useful, consider buying me a coffee! Your support helps maintain and improve the project.

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/yourusername)

## Acknowledgments

This project stands on the shoulders of giants. We'd like to acknowledge the following projects and their contributors:

### Core Technologies
- [OpenAI Whisper](https://github.com/openai/whisper) - The foundation of our speech recognition capabilities
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) - CTranslate2-based Whisper implementation
- [FFmpeg](https://ffmpeg.org/) - The backbone of our audio/video processing
- [PyTorch](https://pytorch.org/) - Deep learning framework powering Whisper

### Subtitle Processing
- [Subliminal](https://github.com/Diaoul/subliminal) - Subtitle downloading and processing
- [OpenSubtitles](https://www.opensubtitles.org/) - Subtitle database and API
- [pysrt](https://github.com/byroot/pysrt) - SRT subtitle parsing and manipulation

### Machine Learning Acceleration
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) - GPU acceleration for NVIDIA hardware
- [Apple CoreML](https://developer.apple.com/documentation/coreml) - Neural Engine acceleration for Apple Silicon

### Python Libraries
- [tqdm](https://github.com/tqdm/tqdm) - Progress bar functionality
- [chardet](https://github.com/chardet/chardet) - Character encoding detection
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) - Python bindings for FFmpeg
- [babelfish](https://github.com/Diaoul/babelfish) - Language code handling

### Docker Support
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) - GPU support in containers
- [Docker Buildx](https://github.com/docker/buildx) - Multi-platform build support

### Inspiration
- [CleanVid](https://github.com/clean-vid) - Inspiration for subtitle-based censoring approach
- [profanity-filter](https://github.com/rominf/profanity-filter) - Profanity detection techniques

Special thanks to all the maintainers and contributors of these projects who make open source amazing!

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate and follow the existing code style.

---
Made with ❤️ by [https://buymeacoffee.com/smukoti] 