# CensorBot

[![CI/CD](https://github.com/samuelmukoti/censorbot/actions/workflows/ci.yml/badge.svg)](https://github.com/samuelmukoti/censorbot/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/censorbot.svg)](https://pypi.org/project/censorbot/)
[![Docker Pulls](https://img.shields.io/docker/pulls/samuelmukoti/censorbot.svg)](https://hub.docker.com/r/samuelmukoti/censorbot)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-donate-yellow.svg)](https://www.buymeacoffee.com/smukoti)

A powerful Python-based tool for automatically censoring profanity in video files. CensorBot uses a multi-stage approach to detect and censor inappropriate language, combining embedded subtitles, online subtitle databases, and AI-powered transcription to ensure accurate profanity detection. Perfect for making your Blu-ray collection, streaming content, or personal video library family-friendly and suitable for all audiences.

## ⚠️ Legal Disclaimer

**CensorBot is intended for personal, educational, and lawful use only.**

- **Media Ownership Required**: You must legally own or have proper authorization to use any media processed with this tool. CensorBot is designed for personal media libraries, legally purchased content, and authorized educational materials.
- **No Endorsement of Piracy**: This project and its author(s) **do not endorse, promote, or support piracy** or copyright infringement in any form.
- **User Responsibility**: Users are solely responsible for ensuring they have the legal right to modify and use any media files processed with CensorBot. This includes compliance with copyright laws, terms of service, and licensing agreements.
- **No Warranty**: This software is provided "as is" without warranty of any kind. The author(s) are not liable for any misuse, legal consequences, or damages arising from the use of this tool.

By using CensorBot, you acknowledge that you understand and agree to these terms, and that you will use the software responsibly and in accordance with applicable laws.

[!NOTE]
## Why Use CensorBot?
CensorBot is designed for users who want to make their video content family-friendly, educational, or suitable for public viewing by automatically removing or masking profane language. Here are some common scenarios:

- **Family Movie Nights**: Make your Blu-ray or digital movie collection safe for children by muting or beeping out offensive words.
- **Classroom/Educational Use**: Teachers can use CensorBot to prepare video materials for classroom use, ensuring compliance with school policies.
- **Streaming/Content Creation**: Streamers and YouTubers can quickly sanitize videos before publishing to avoid demonetization or content strikes.
- **Community Events**: Organizers can prepare movies for public screenings in community centers, churches, or youth groups.
- **Corporate Training**: HR teams can remove inappropriate language from training videos for workplace compliance.

## How to Use CensorBot
1. **Select Your Video**: Choose the video file you want to censor (MP4, MKV, AVI supported).
2. **Choose Censoring Mode**: Decide whether you want to mute, beep, or keep both original and censored audio tracks.
3. **Customize Wordlist**: Optionally provide your own list of words to censor for specific needs.
4. **Run CensorBot**: Use the provided Docker commands to process your video. Example:
  ```bash
  docker run -v $(pwd):/app censorbot -i input.mp4 -o output.mp4 --mode beep
  ```
5. **Review Output**: The output video will have censored audio, ready for safe viewing or sharing.

See the Usage section below for more command examples and options.

## Recent Updates (2024 Rewrite)

This project has been completely rewritten with significant improvements:
- ✅ **Simplified FFmpeg filtering**: Replaced 382 lines of broken batching logic with clean chained filters
- ✅ **Real hardware acceleration**: Added MLX support for Apple Silicon (Metal/Neural Engine)
- ✅ **Automatic fallback**: MLX → faster-whisper CPU for robust operation
- ✅ **Production-ready**: Tested on full-length movies with verified results
- ✅ **Cleaner architecture**: Removed unused code, improved error handling, modern dependencies

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
  - NVIDIA GPUs: CUDA acceleration via faster-whisper
  - Apple Silicon: MLX framework with Metal and Neural Engine acceleration
  - Intel/AMD: Multi-threaded CPU processing with int8 quantization
- Automatic fallback mechanism (MLX → faster-whisper CPU)
- Real-time progress tracking for transcription operations
- Efficient memory management with temporary file cleanup

### NEW in v2.0.0 🎉
- **Dry-Run Mode**: Preview what will be censored before processing (`--dry-run`)
- **Export Censored Subtitles**: Generate SRT files with profanity replaced (`--export-srt`)
- **Word Statistics**: See detailed profanity reports before censoring (`--stats`)
- **Custom Beep Sounds**: Use your own audio file for beep mode (`--beep-file`)
- **Configuration Files**: Save settings in YAML for repeated use (`--config`)
- **Progress Bars**: Visual feedback during long transcription operations
- **Pip Installation**: Now available via `pip install censorbot`

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

### Option 1: Pip Install (Recommended)

```bash
# Install from PyPI
pip install censorbot

# Install with Apple Silicon MLX support
pip install censorbot[mlx]  # macOS ARM64 only

# Run censorbot
censorbot -i input.mp4 -o output.mp4
```

**Note**: Requires FFmpeg to be installed separately:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### Option 2: Docker (Isolated Environment)

1. Clone the repository:
```bash
git clone https://github.com/samuelmukoti/censorbot.git
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
docker run -v $(pwd):/app censorbot -i input.mp4 -o output.mp4
```

### Platform-Specific Usage

1. **NVIDIA GPU (Linux/Windows)**
```bash
docker run --gpus all -v $(pwd):/app censorbot -i input.mp4 -o output.mp4
```

2. **Apple Silicon (M1/M2/M3 Mac)**
```bash
# Automatically uses MLX (Metal acceleration) with fallback to CPU
docker run -v $(pwd):/app censorbot -i input.mp4 -o output.mp4
```

3. **Force CPU Processing (Any Platform)**
```bash
docker run -v $(pwd):/app censorbot -i input.mp4 -o output.mp4 --force-cpu
```

### Common Use Cases

1. **Using Beep Sound Instead of Muting**
```bash
docker run -v $(pwd):/app censorbot -i input.mp4 -o output.mp4 --mode beep
```

2. **Using Custom Subtitle File**
```bash
docker run -v $(pwd):/app censorbot -i input.mp4 -o output.mp4 -s subtitles.srt
```

3. **Single Audio Track (Censored Only)**
```bash
docker run -v $(pwd):/app censorbot -i input.mp4 -o output.mp4 --single-audio
```

4. **Custom Wordlist**
```bash
docker run -v $(pwd):/app censorbot -i input.mp4 -o output.mp4 -w custom_badwords.txt
```

### Advanced Options

#### Command Line Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `-i, --input` | Input video file path | Required |
| `-o, --output` | Output video file path | Required |
| `-w, --wordlist` | Custom wordlist file | Built-in list (39 words) |
| `-s, --subtitles` | External subtitle file (SRT) | None |
| `--force-cpu` | Disable hardware acceleration | False |
| `--model-size` | Whisper model size (tiny/base/small/medium/large) | base |
| `--padding` | Padding around censored segments (seconds) | 0.2 |
| `--mode` | Censoring mode (mute/beep) | mute |
| `--single-audio` | Only keep censored audio track | False |

## Performance Optimization

### Platform-Specific Acceleration

#### NVIDIA GPUs
- Ensure NVIDIA drivers and Container Toolkit are installed
- Use `--gpus all` flag when running Docker
- Uses CUDA acceleration via faster-whisper
- Adjust model size based on available VRAM (base model ~1GB)

#### Apple Silicon (M1/M2/M3)
- Automatically detects Apple Silicon and attempts MLX acceleration
- Falls back to faster-whisper CPU if MLX fails
- No additional flags needed
- Optimized for efficiency and battery life

#### CPU-Only Systems
- Uses faster-whisper with multi-threading
- Int8 quantization for reduced memory usage
- Automatically uses all available CPU cores
- Consider using smaller model sizes (tiny/base) for faster processing

### Transcription Performance
Expected transcription times (base model):
- **2-hour movie**:
  - Apple Silicon (MLX): ~15-20 minutes
  - CPU (faster-whisper): ~35-45 minutes
  - NVIDIA GPU (CUDA): ~10-15 minutes

## Troubleshooting

### Platform-Specific Issues

1. **NVIDIA GPU Issues**
   - Verify NVIDIA drivers are installed: `nvidia-smi`
   - Check NVIDIA Container Toolkit installation
   - Ensure `--gpus all` flag is used when running Docker
   - Monitor GPU memory usage

2. **Apple Silicon Issues**
   - Ensure using ARM64 version of Docker
   - MLX acceleration may fail if model unavailable (automatic fallback to CPU)
   - Monitor system temperature during long transcriptions
   - Consider `--force-cpu` if experiencing thermal throttling

3. **General Performance Issues**
   - Check logs for acceleration backend: "Using Apple Silicon Metal acceleration (MLX)" or "Using NVIDIA CUDA acceleration"
   - Monitor system resources (CPU, memory) during transcription
   - Adjust model size (tiny/base for faster, medium/large for accuracy)
   - Large files may take significant time (35-45 min for 2-hour movie on CPU)

### Common Workflow Messages

**Expected behavior (not errors):**
- `No subtitles found from online sources`: Normal - will fallback to AI transcription
- `MLX transcription failed`: Normal - automatically falls back to faster-whisper CPU
- `Repository Not Found` for MLX models: Normal - fallback mechanism handles this
- Runtime warnings during transcription: Non-critical numerical artifacts

**Actual errors:**
- `Failed to extract audio`: Check video file format and permissions
- `FFmpeg error`: Ensure FFmpeg is installed and video file is not corrupted
- `Out of memory`: Reduce model size (use `--model-size tiny` or `--model-size base`)
- Subtitle provider errors: Expected when providers are unavailable or require auth

## Example Results

**Test case:  1080p Blu-Ray movie**
- **File size**: 2.6GB
- **Duration**: 2:13:59
- **Processing time**: ~42 minutes (Apple Silicon M-series, CPU fallback)
  - Transcription: 38 minutes
  - Audio censoring: 26 seconds
  - Video merging: 3 minutes 52 seconds
- **Results**:
  - ✅ Transcribed 12,635 words
  - ✅ Found 181 profane words to censor
  - ✅ Applied 181 censorship segments
  - ✅ Output: Dual-audio MP4 (original + censored tracks)
  - ✅ Video quality preserved (no re-encoding)

## Frequently Asked Questions (FAQ)

### General Questions

**Q: Why is transcription so slow?**
A: AI transcription is computationally intensive. A 2-hour movie takes 35-45 minutes on CPU, 15-20 minutes with MLX (Apple Silicon), or 10-15 minutes with NVIDIA GPU. Use `--dry-run` to preview without full processing, or provide subtitle files with `-s` to skip transcription entirely.

**Q: Can I use my own wordlist?**
A: Yes! Use `-w custom_words.txt` with one word per line. The default list has 39 common profanities. Your custom list will be combined with the default unless you modify the code.

**Q: How do I switch between original and censored audio?**
A: By default, the output has two audio tracks. In VLC: Audio → Audio Track → Track 1 (original) or Track 2 (censored). Use `--single-audio` to keep only the censored track.

**Q: Can I preview what will be censored without processing the whole video?**
A: Yes! Use `--dry-run` to see timestamps and profane words that would be censored:
```bash
censorbot -i video.mp4 -o output.mp4 --dry-run --stats
```

**Q: Does this work on streaming content (Netflix, YouTube, etc.)?**
A: No. CensorBot requires downloadable video files (MP4, MKV, AVI). Use screen recording tools first, then process the recording.

### Technical Questions

**Q: Do I need an NVIDIA GPU or Apple Silicon?**
A: No. CensorBot works on any system with CPU-only mode (slower). GPU/MLX acceleration is optional for faster processing.

**Q: Why did subtitle download fail?**
A: This is expected for many videos. OpenSubtitles requires authentication, and providers may time out. CensorBot automatically falls back to AI transcription when subtitles aren't available.

**Q: Can I use this in a script or automation?**
A: Yes! Use configuration files for consistent settings:
```yaml
# config.yaml
mode: mute
model: base
padding: 0.2
stats: true
```
Then run: `censorbot --config config.yaml -i video.mp4 -o output.mp4`

**Q: How accurate is the profanity detection?**
A: Very accurate with subtitles (near 100%). With AI transcription, accuracy depends on audio quality and accents (typically 85-95% for clear English audio).

**Q: Can I censor specific words only?**
A: Yes. Create a custom wordlist with only the words you want censored and use `-w your_words.txt`.

**Q: Will this work for languages other than English?**
A: The current implementation is optimized for English. Whisper supports 90+ languages, but you'd need to provide language-specific wordlists.

### Installation & Setup

**Q: Do I need to install Docker?**
A: Not anymore! Install via pip: `pip install censorbot`. Docker is optional for isolated environments.

**Q: I get "FFmpeg not found" error**
A: Install FFmpeg separately:
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt-get install ffmpeg`
- Windows: Download from ffmpeg.org

**Q: How do I install on Apple Silicon (M1/M2/M3)?**
A: ```bash
pip install censorbot[mlx]
```
This includes MLX for Metal acceleration (5-10x faster than CPU).

### Performance & Optimization

**Q: Can I make it faster?**
A:
1. Provide subtitle files with `-s subtitles.srt` (skips transcription)
2. Use smaller Whisper model: `--model-size tiny` (faster but less accurate)
3. Use GPU/MLX acceleration if available
4. Use `--dry-run` for testing without full processing

**Q: How much disk space do I need?**
A: Temporary files during processing require ~2x the input video size. Final output is similar to input size.

**Q: Why is my output video file size different?**
A: The video stream is copied (not re-encoded), but audio is re-encoded. Dual-audio outputs are ~5-10% larger. Use `--single-audio` for smaller files.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Support the Project

If you find this tool useful, consider buying me a coffee! Your support helps maintain and improve the project.

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/smukoti)

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
- [Apple MLX](https://github.com/ml-explore/mlx) - Metal and Neural Engine acceleration for Apple Silicon
- [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) - Whisper optimized for Apple Silicon

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