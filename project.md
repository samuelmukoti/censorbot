Project: Audio Censorship Tool for Video Files
Objective: Create a Dockerized Python tool that automatically censors curse words in video files by muting audio segments, offering both subtitle-based and AI transcription approaches.

Core Features
Input/Output Handling

Accept MP4, MKV, AVI video inputs

Output MP4 with original + censored audio tracks

Two Censoring Modes

Mode 1: Subtitle-based (Cleanvid-style)

Accept SRT/ASS subtitle files

Mute audio where curse words appear in subtitles

Mode 2: AI Transcription (Whisper-based)

Auto-generate word-level timestamps using Faster-Whisper

Fallback to subtitle mode if subtitles exist

Audio Processing

Mute detected segments with configurable padding (±0.2s default)

Preserve original audio quality (lossless processing)

Add censored track as secondary audio stream

Configuration

Curse word list (text file with 1 word per line)

Toggle between silence/beep censoring

GPU acceleration flag for Whisper

Technical Requirements
Component	Tools/Stack	Key Requirements
Audio Extraction	FFmpeg	Handle multiple audio codecs
Speech Recognition	Faster-Whisper (CUDA)	Word-level timestamps with <100ms variance
Text Matching	Custom regex + spaCy	Handle case variations (e.g., "Sh!t")
Docker Environment	NVIDIA CUDA base image	<2GB final image size
Performance		<10min processing for 1hr video (GPU)
Deliverables
Code

Dockerfile with preloaded Whisper models

Python script (censor.py) with CLI interface

Sample curse words list (badwords.txt)

Documentation

Usage examples:

bash
Copy
# Subtitle mode
docker run -v /videos:/app censortool -i input.mp4 -s subs.srt -w badwords.txt

# AI mode (GPU)
docker run --gpus all -v /videos:/app censortool -i input.mp4 -w badwords.txt
Benchmark results for different hardware

QA Artifacts

Unit tests for timestamp alignment

Sample videos with known curse word patterns

Validation script to check mute intervals

Milestones & Timeline
Environment Setup (2 Days)

Docker image with FFmpeg + Whisper

Basic audio extraction/merging

Core Features (5 Days)

Subtitle-based censoring

Whisper transcription pipeline

Audio muting logic

Optimization (3 Days)

GPU acceleration

Parallel processing for long videos

Testing (2 Days)

Edge cases (overlapping words, low-quality audio)

Format compatibility check

Key Dependencies
Third-Party

FFmpeg 6.0+

NVIDIA CUDA 12.x (for GPU mode)

Pre-trained Whisper models

Assumptions

Curse word list provided by user

Input videos contain at least 1 audio track

Risks & Mitigation
Risk	Mitigation
Whisper timestamp inaccuracy	Configurable padding (+/- 0.5s max)
GPU memory constraints	Auto-switch to CPU mode with warning
Subtitle/audio misalignment	Add validation check with pysrt library

