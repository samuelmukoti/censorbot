#!/usr/bin/env python3
"""
CensorBot - Video Profanity Censoring Tool
Automatically detects and censors profanity in video files using:
1. Embedded subtitles extraction
2. Online subtitle download (OpenSubtitles)
3. AI transcription (Whisper with hardware acceleration)
"""

import argparse
import os
import sys
import logging
import subprocess
import json
import tempfile
import re
import platform
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor

import pysrt
import chardet
import torch
from tqdm import tqdm
from subliminal import download_best_subtitles, region, save_subtitles, scan_video
from babelfish import Language

# Whisper backend imports
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

# MLX backend for Apple Silicon
MLX_AVAILABLE = False
if platform.processor() == 'arm' and platform.system() == 'Darwin':
    try:
        import mlx_whisper
        MLX_AVAILABLE = True
    except ImportError:
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default profanity list
DEFAULT_BADWORDS = {
    # Common profanity
    "fuck", "shit", "ass", "bitch", "damn", "hell",
    # Racial slurs
    "nigger", "nigga", "chink", "spic", "wetback", "kike", "gook",
    # Sexual terms
    "cock", "dick", "pussy", "cunt", "whore", "slut",
    # Religious profanity
    "goddamn", "jesus christ",
    # Variations and misspellings
    "f*ck", "sh*t", "b*tch", "f**k", "s**t", "fuk", "fck", "fucker",
    # Compound words
    "motherfucker", "motherfucking", "fucking", "bullshit", "horseshit",
    "asshole", "dumbass", "jackass", "dipshit", "douchebag"
}


class CensorBot:
    """Main class for video profanity censoring with hardware acceleration."""

    def __init__(self, model_size: str = "base", use_gpu: bool = True, force_cpu: bool = False):
        """Initialize CensorBot with appropriate hardware acceleration.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            use_gpu: Enable GPU acceleration if available
            force_cpu: Force CPU-only processing
        """
        self.model_size = model_size
        self.setup_acceleration(model_size, use_gpu, force_cpu)

        # Initialize subtitle cache
        region.configure('dogpile.cache.dbm', arguments={'filename': 'cachefile.dbm'})

        # Create beep sound file
        self._create_beep_file()

    def setup_acceleration(self, model_size: str, use_gpu: bool, force_cpu: bool):
        """Setup hardware acceleration based on platform and available hardware.

        Supports:
        - Apple Silicon: MLX with Metal acceleration
        - NVIDIA GPUs: CUDA acceleration
        - CPU: Multi-threaded processing
        """
        if force_cpu:
            logger.info("Forcing CPU-only processing")
            self._setup_cpu_backend(model_size)
            return

        # Try MLX for Apple Silicon first
        if MLX_AVAILABLE and use_gpu:
            logger.info("✓ Using Apple Silicon Metal acceleration (MLX)")
            self.backend = 'mlx'
            self.model = None  # MLX uses functional API
            return

        # Try CUDA for NVIDIA GPUs
        if FASTER_WHISPER_AVAILABLE and torch.cuda.is_available() and use_gpu:
            logger.info("✓ Using NVIDIA CUDA acceleration")
            self.model = WhisperModel(
                model_size,
                device="cuda",
                compute_type="float16"
            )
            self.backend = 'cuda'
            return

        # Fallback to CPU
        self._setup_cpu_backend(model_size)

    def _setup_cpu_backend(self, model_size: str):
        """Setup CPU-only processing."""
        if not FASTER_WHISPER_AVAILABLE:
            logger.warning("⚠ Faster-Whisper not available. Transcription will be disabled.")
            self.backend = 'none'
            self.model = None
            return

        logger.info("Using CPU multi-threaded processing")
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=os.cpu_count()
        )
        self.backend = 'cpu'

    def _create_beep_file(self):
        """Create a beep sound file using FFmpeg."""
        self.beep_file = "beep.wav"
        if not os.path.exists(self.beep_file):
            try:
                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi",
                    "-i", "sine=frequency=1000:duration=0.5",
                    "-ar", "16000",
                    self.beep_file
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                logger.debug(f"Created beep file: {self.beep_file}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to create beep file: {e}")

    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file in seconds using FFprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.error(f"Failed to get audio duration: {e}")
            return 0.0

    def extract_embedded_subtitles(self, video_path: str) -> Optional[str]:
        """Extract embedded subtitles from video file if they exist.

        Args:
            video_path: Path to input video file

        Returns:
            Path to extracted subtitle file or None if not found
        """
        try:
            temp_srt = tempfile.NamedTemporaryFile(suffix='.srt', delete=False)
            temp_srt.close()

            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", video_path,
                "-map", "0:s:0",
                "-f", "srt",
                temp_srt.name
            ]

            subprocess.run(cmd, check=True, capture_output=True)

            if os.path.exists(temp_srt.name) and os.path.getsize(temp_srt.name) > 0:
                logger.info("✓ Successfully extracted embedded subtitles")
                return temp_srt.name
            else:
                os.unlink(temp_srt.name)
                return None

        except subprocess.CalledProcessError:
            logger.debug("No embedded subtitles found")
            if os.path.exists(temp_srt.name):
                os.unlink(temp_srt.name)
            return None

    def download_subtitles(self, video_path: str) -> Optional[str]:
        """Download subtitles from OpenSubtitles.

        Args:
            video_path: Path to input video file

        Returns:
            Path to downloaded subtitle file or None if not found
        """
        try:
            logger.info("Attempting to download subtitles from OpenSubtitles...")
            video = scan_video(video_path)

            # Configure providers with credentials if available
            provider_configs = {}
            if os.environ.get('OPENSUBTITLES_USERNAME'):
                provider_configs['opensubtitles'] = {
                    'username': os.environ.get('OPENSUBTITLES_USERNAME'),
                    'password': os.environ.get('OPENSUBTITLES_PASSWORD')
                }

            subtitles = download_best_subtitles(
                [video],
                {Language('eng')},
                provider_configs=provider_configs
            )

            if subtitles.get(video):
                subtitle = list(subtitles[video])[0]
                temp_srt = tempfile.NamedTemporaryFile(suffix='.srt', delete=False)
                temp_srt.close()

                save_subtitles(video, [subtitle], encoding='utf-8')
                logger.info("✓ Successfully downloaded subtitles")
                return temp_srt.name
            else:
                logger.warning("No subtitles found from online sources")
                return None

        except Exception as e:
            logger.warning(f"Failed to download subtitles: {e}")
            return None

    def extract_audio(self, video_path: str, output_path: str) -> bool:
        """Extract audio from video file using FFmpeg.

        Args:
            video_path: Path to input video file
            output_path: Path for extracted audio file

        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"✓ Extracted audio to {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract audio: {e}")
            return False

    def load_badwords(self, badwords_file: Optional[str] = None) -> List[str]:
        """Load bad words from file or use defaults.

        Args:
            badwords_file: Optional path to custom badwords file

        Returns:
            List of bad words to censor
        """
        if badwords_file and os.path.exists(badwords_file):
            try:
                with open(badwords_file, 'r', encoding='utf-8') as f:
                    custom_words = {word.strip().lower() for word in f if word.strip()}
                words = list(DEFAULT_BADWORDS.union(custom_words))
                logger.info(f"Loaded {len(words)} words to censor ({len(custom_words)} custom)")
                return words
            except Exception as e:
                logger.warning(f"Failed to load custom badwords file: {e}. Using defaults.")

        logger.info(f"Using default badwords list ({len(DEFAULT_BADWORDS)} words)")
        return list(DEFAULT_BADWORDS)

    def process_subtitles(self, srt_file: str, badwords: List[str]) -> Tuple[List[Tuple[float, float]], str]:
        """Process SRT file to find timestamps of bad words and censor subtitle text.

        Args:
            srt_file: Path to subtitle file
            badwords: List of words to censor

        Returns:
            Tuple of (censor_segments, censored_subtitle_path)
        """
        try:
            # Detect encoding
            with open(srt_file, 'rb') as f:
                raw_data = f.read()
                encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'

            # Load subtitles
            subs = pysrt.open(srt_file, encoding=encoding)
            censor_segments = []

            # Create censored subtitle file
            temp_srt = tempfile.NamedTemporaryFile(suffix='.srt', delete=False)
            temp_srt.close()

            # Compile regex patterns with word boundaries
            patterns = [re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                       for word in badwords]

            for sub in subs:
                text = sub.text
                has_profanity = False

                # Check and censor each bad word
                for pattern in patterns:
                    if pattern.search(text):
                        has_profanity = True
                        # Replace with asterisks
                        text = pattern.sub(lambda m: '*' * len(m.group()), text)

                if has_profanity:
                    # Convert timestamp to seconds
                    start = (sub.start.hours * 3600 + sub.start.minutes * 60 +
                            sub.start.seconds + sub.start.milliseconds / 1000.0)
                    end = (sub.end.hours * 3600 + sub.end.minutes * 60 +
                          sub.end.seconds + sub.end.milliseconds / 1000.0)
                    censor_segments.append((start, end))

                sub.text = text

            # Save censored subtitles
            subs.save(temp_srt.name, encoding='utf-8')
            logger.info(f"✓ Processed subtitles: found {len(censor_segments)} segments to censor")
            return censor_segments, temp_srt.name

        except Exception as e:
            logger.error(f"Failed to process subtitles: {e}")
            return [], None

    def transcribe_audio(self, audio_path: str) -> List[Tuple[float, float, str]]:
        """Transcribe audio using Whisper with hardware acceleration.

        Args:
            audio_path: Path to audio file

        Returns:
            List of (start_time, end_time, word) tuples
        """
        if self.backend == 'none':
            raise RuntimeError("No transcription backend available")

        logger.info(f"Transcribing audio using {self.backend.upper()} backend...")

        if self.backend == 'mlx':
            try:
                # Use MLX backend for Apple Silicon
                result = mlx_whisper.transcribe(
                    audio_path,
                    path_or_hf_repo=f"mlx-community/whisper-{self.model_size}",
                    word_timestamps=True
                )

                word_timestamps = []
                segments_list = list(result.get('segments', []))
                for segment in tqdm(segments_list, desc="Transcribing", unit="segment"):
                    for word_info in segment.get('words', []):
                        word_timestamps.append((
                            word_info['start'],
                            word_info['end'],
                            word_info['word'].strip().lower()
                        ))

            except Exception as e:
                logger.warning(f"MLX transcription failed: {e}")
                logger.info("Falling back to faster-whisper CPU backend...")

                # Fallback to faster-whisper CPU
                if not hasattr(self, 'model') or self.model is None:
                    from faster_whisper import WhisperModel
                    self.model = WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=os.cpu_count()
                    )

                segments, _ = self.model.transcribe(
                    audio_path,
                    word_timestamps=True,
                    language="en"
                )

                word_timestamps = []
                segments_list = list(segments)
                for segment in tqdm(segments_list, desc="Transcribing", unit="segment"):
                    if hasattr(segment, 'words'):
                        for word in segment.words:
                            word_timestamps.append((
                                word.start,
                                word.end,
                                word.word.strip().lower()
                            ))

        else:
            # Use faster-whisper backend (CUDA or CPU)
            segments, _ = self.model.transcribe(
                audio_path,
                word_timestamps=True,
                language="en"
            )

            word_timestamps = []
            segments_list = list(segments)
            for segment in tqdm(segments_list, desc="Transcribing", unit="segment"):
                if hasattr(segment, 'words'):
                    for word in segment.words:
                        word_timestamps.append((
                            word.start,
                            word.end,
                            word.word.strip().lower()
                        ))

        logger.info(f"✓ Transcribed {len(word_timestamps)} words")
        return word_timestamps

    def find_censor_segments(self,
                           transcription: List[Tuple[float, float, str]],
                           badwords: List[str],
                           padding: float = 0.2) -> List[Tuple[float, float]]:
        """Find segments to censor based on transcription and bad words.

        Args:
            transcription: List of (start, end, word) tuples
            badwords: List of words to censor
            padding: Padding in seconds before/after each word

        Returns:
            List of (start, end) tuples for segments to censor
        """
        censor_segments = []
        badwords_set = set(word.lower() for word in badwords)

        for start, end, word in transcription:
            word_clean = word.strip().lower()
            # Check exact match or if word contains badword
            if word_clean in badwords_set or any(bad in word_clean for bad in badwords_set):
                censor_segments.append((
                    max(0, start - padding),
                    end + padding
                ))

        logger.info(f"✓ Found {len(censor_segments)} profane words to censor")
        return censor_segments

    def export_censored_subtitles(self,
                                 transcription: List[Tuple[float, float, str]],
                                 badwords: List[str],
                                 output_path: str) -> bool:
        """Export subtitles with profane words replaced by [censored].

        Args:
            transcription: List of (start, end, word) tuples
            badwords: List of words to censor
            output_path: Path for output SRT file

        Returns:
            True if successful, False otherwise
        """
        try:
            badwords_set = set(word.lower() for word in badwords)
            subs = pysrt.SubRipFile()

            # Group words into subtitle segments (roughly 5-second chunks)
            current_segment = []
            segment_start = None
            index = 1

            for i, (start, end, word) in enumerate(transcription):
                if not current_segment:
                    segment_start = start

                word_clean = word.strip().lower()
                # Check if word should be censored
                if word_clean in badwords_set or any(bad in word_clean for bad in badwords_set):
                    current_segment.append('[censored]')
                else:
                    current_segment.append(word.strip())

                # Create subtitle when segment reaches ~5 seconds or end of transcription
                if (start - segment_start >= 5.0) or (i == len(transcription) - 1):
                    if current_segment:
                        # Convert timestamps to SubRipTime
                        start_time = pysrt.SubRipTime(seconds=segment_start)
                        end_time = pysrt.SubRipTime(seconds=end)

                        # Create subtitle item
                        sub = pysrt.SubRipItem(
                            index=index,
                            start=start_time,
                            end=end_time,
                            text=' '.join(current_segment)
                        )
                        subs.append(sub)
                        index += 1

                    # Reset for next segment
                    current_segment = []
                    segment_start = None

            # Save to file
            subs.save(output_path, encoding='utf-8')
            logger.info(f"✓ Exported censored subtitles to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export censored subtitles: {e}")
            return False

    def print_statistics(self,
                        censor_segments: List[Tuple[float, float]],
                        transcription: List[Tuple[float, float, str]],
                        badwords: List[str]) -> None:
        """Print statistics about censored content.

        Args:
            censor_segments: List of (start, end) tuples for censored segments
            transcription: List of (start, end, word) tuples from transcription
            badwords: List of bad words to check against
        """
        badwords_set = set(word.lower() for word in badwords)
        total_words = len(transcription)
        profane_words = []

        # Find all profane words
        for start, end, word in transcription:
            word_clean = word.strip().lower()
            if word_clean in badwords_set or any(bad in word_clean for bad in badwords_set):
                profane_words.append(word_clean)

        # Calculate statistics
        num_profane = len(profane_words)
        percentage = (num_profane / total_words * 100) if total_words > 0 else 0

        # Count word frequencies
        from collections import Counter
        word_counts = Counter(profane_words)
        top_5 = word_counts.most_common(5)

        # Print report
        print("\n" + "=" * 60)
        print("CENSORSHIP STATISTICS")
        print("=" * 60)
        print(f"Total words transcribed: {total_words}")
        print(f"Profane words found: {num_profane}")
        print(f"Profanity percentage: {percentage:.2f}%")
        print(f"Segments to censor: {len(censor_segments)}")

        if top_5:
            print("\nTop 5 most frequent profane words:")
            for i, (word, count) in enumerate(top_5, 1):
                print(f"  {i}. '{word}': {count} occurrence(s)")

        print("=" * 60 + "\n")

    def apply_censorship(self,
                        input_audio: str,
                        output_audio: str,
                        censor_segments: List[Tuple[float, float]],
                        mode: str = 'mute',
                        beep_file: Optional[str] = None) -> bool:
        """Apply audio censorship using simplified FFmpeg filter chains.

        This is a complete rewrite that fixes all the batching/complexity issues.
        Uses simple chained volume filters for accurate, clean censoring.

        Args:
            input_audio: Path to input audio file
            output_audio: Path for output audio file
            censor_segments: List of (start, end) tuples to censor
            mode: 'mute' (silence) or 'beep' (replace with beep)
            beep_file: Optional custom beep sound file (for beep mode)

        Returns:
            True if successful, False otherwise
        """
        try:
            if not censor_segments:
                logger.info("No segments to censor, copying audio as-is")
                cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", input_audio,
                       "-c:a", "copy", output_audio]
                subprocess.run(cmd, check=True, capture_output=True)
                return True

            logger.info(f"Censoring {len(censor_segments)} segments in {mode} mode")

            if mode == 'mute':
                # Simple mute mode: chain volume filters
                volume_filters = [
                    f"volume=enable='between(t,{start},{end})':volume=0"
                    for start, end in censor_segments
                ]

                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", input_audio,
                    "-af", ','.join(volume_filters),
                    output_audio
                ]

            elif mode == 'beep':
                # Beep mode: create continuous sine or use custom beep, enable at censor points, mix with muted
                duration = self.get_audio_duration(input_audio)

                # Create volume enable expressions for beeps
                beep_enables = ','.join([
                    f"volume=enable='between(t,{start},{end})':volume=0.3"
                    for start, end in censor_segments
                ])

                # Create mute filters
                mute_filters = ','.join([
                    f"volume=enable='between(t,{start},{end})':volume=0"
                    for start, end in censor_segments
                ])

                # Build filter complex - use custom beep file or generate sine wave
                if beep_file and os.path.exists(beep_file):
                    logger.info(f"Using custom beep file: {beep_file}")
                    # Use custom beep file - loop it to match duration
                    filter_complex = (
                        f"[1:a]aloop=loop=-1:size=2e+09,atrim=0:{duration}[beep_loop];"
                        f"[beep_loop]{beep_enables}[beep];"
                        f"[0:a]{mute_filters}[muted];"
                        f"[muted][beep]amix=inputs=2:duration=first[out]"
                    )
                    cmd = [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-i", input_audio,
                        "-i", beep_file,
                        "-filter_complex", filter_complex,
                        "-map", "[out]",
                        output_audio
                    ]
                else:
                    # Generate sine wave
                    filter_complex = (
                        f"sine=frequency=1000:duration={duration}:sample_rate=16000[sine];"
                        f"[sine]{beep_enables}[beep];"
                        f"[0:a]{mute_filters}[muted];"
                        f"[muted][beep]amix=inputs=2:duration=first[out]"
                    )
                    cmd = [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-i", input_audio,
                        "-filter_complex", filter_complex,
                        "-map", "[out]",
                        output_audio
                    ]

            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'mute' or 'beep'")

            # Execute FFmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False

            logger.info("✓ Successfully applied audio censorship")
            return True

        except Exception as e:
            logger.error(f"Failed to apply censorship: {e}")
            return False

    def merge_audio_video(self,
                         video_path: str,
                         censored_audio_path: str,
                         output_path: str,
                         dual_audio: bool = True) -> bool:
        """Merge video with censored audio, optionally keeping both audio tracks.

        Args:
            video_path: Path to original video (with original audio)
            censored_audio_path: Path to censored audio file
            output_path: Path for output video
            dual_audio: If True, keep both original and censored audio tracks

        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
                   "-i", censored_audio_path]

            if dual_audio:
                # Keep both original and censored audio as separate tracks
                cmd.extend([
                    "-map", "0:v",  # Video from original
                    "-map", "0:a",  # Original audio
                    "-map", "1:a",  # Censored audio
                    "-c:v", "copy",  # Copy video (no re-encode)
                    "-c:a:0", "aac", "-b:a:0", "192k",  # Original audio track
                    "-c:a:1", "aac", "-b:a:1", "192k",  # Censored audio track
                    "-metadata:s:a:0", "title=Original Audio",
                    "-metadata:s:a:1", "title=Censored Audio"
                ])
            else:
                # Only use censored audio
                cmd.extend([
                    "-map", "0:v",  # Video from original
                    "-map", "1:a",  # Censored audio only
                    "-c:v", "copy",  # Copy video (no re-encode)
                    "-c:a", "aac", "-b:a", "192k"
                ])

            cmd.append(output_path)

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False

            logger.info("✓ Successfully merged audio and video")
            return True

        except Exception as e:
            logger.error(f"Failed to merge audio and video: {e}")
            return False

    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Dictionary of configuration values
        """
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✓ Loaded configuration from {config_path}")
            return config or {}
        except ImportError:
            logger.error("PyYAML not installed. Install with: pip install pyyaml")
            return {}
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            return {}

    def embed_subtitles(self,
                       video_path: str,
                       subtitle_path: str,
                       output_path: str) -> bool:
        """Embed subtitles into the video file while preserving all audio tracks.

        Args:
            video_path: Path to input video
            subtitle_path: Path to subtitle file
            output_path: Path for output video

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get number of audio streams
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                video_path
            ]
            audio_streams = subprocess.check_output(probe_cmd).decode().strip().split('\n')
            num_audio = len([s for s in audio_streams if s])

            # Build FFmpeg command
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", video_path,
                "-i", subtitle_path,
                "-map", "0:v",
                "-map", "0:a",
                "-map", "1:0",
                "-c:v", "copy",
                "-c:a", "copy",
                "-c:s", "mov_text",
                "-metadata:s:s:0", "language=eng",
                "-metadata:s:s:0", "title=Censored Subtitles"
            ]

            # Add metadata for audio tracks
            if num_audio >= 2:
                cmd.extend([
                    "-metadata:s:a:0", "title=Original Audio",
                    "-metadata:s:a:1", "title=Censored Audio"
                ])

            cmd.append(output_path)

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False

            logger.info("✓ Successfully embedded subtitles")
            return True

        except Exception as e:
            logger.error(f"Failed to embed subtitles: {e}")
            return False


def process_single_video(input_path: str,
                        output_path: str,
                        badwords_file: Optional[str] = None,
                        subtitles_file: Optional[str] = None,
                        model_size: str = "base",
                        use_gpu: bool = True,
                        force_cpu: bool = False,
                        censor_mode: str = "mute",
                        padding: float = 0.2,
                        dual_audio: bool = True,
                        no_download_subs: bool = False,
                        dry_run: bool = False,
                        export_srt: Optional[str] = None,
                        beep_file: Optional[str] = None,
                        show_stats: bool = False) -> bool:
    """Process a single video file with censoring.

    Args:
        input_path: Path to input video
        output_path: Path for output video
        badwords_file: Optional custom badwords file
        subtitles_file: Optional external subtitle file
        model_size: Whisper model size
        use_gpu: Enable GPU acceleration
        force_cpu: Force CPU-only processing
        censor_mode: 'mute' or 'beep'
        padding: Padding in seconds around censored words
        dual_audio: Keep both original and censored audio tracks
        no_download_subs: Disable automatic subtitle downloading
        dry_run: Preview censorship without creating output
        export_srt: Export censored subtitles to this path
        beep_file: Custom beep sound file for beep mode
        show_stats: Display statistics about censored content

    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize CensorBot
        bot = CensorBot(model_size=model_size, use_gpu=use_gpu, force_cpu=force_cpu)

        # Load badwords
        badwords = bot.load_badwords(badwords_file)

        # Try to get subtitles (3-stage approach)
        subtitle_path = None
        if subtitles_file:
            logger.info(f"Using provided subtitle file: {subtitles_file}")
            subtitle_path = subtitles_file
        else:
            # Try embedded subtitles
            subtitle_path = bot.extract_embedded_subtitles(input_path)

            # Try downloading subtitles
            if not subtitle_path and not no_download_subs:
                subtitle_path = bot.download_subtitles(input_path)

        # Process with temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract audio
            audio_path = os.path.join(temp_dir, "audio.wav")
            if not bot.extract_audio(input_path, audio_path):
                logger.error("Failed to extract audio")
                return False

            # Get censor segments
            censor_segments = []
            censored_subtitle_path = None
            transcription = []

            if subtitle_path:
                logger.info("Using subtitle-based detection")
                censor_segments, censored_subtitle_path = bot.process_subtitles(
                    subtitle_path, badwords
                )
            else:
                logger.info("Using AI transcription (Whisper)")
                transcription = bot.transcribe_audio(audio_path)
                censor_segments = bot.find_censor_segments(
                    transcription, badwords, padding
                )

            if not censor_segments:
                logger.warning("No profanity detected.")
                if dry_run:
                    print("\nDRY RUN: No profanity detected in video.")
                    return True
                import shutil
                shutil.copy2(input_path, output_path)
                return True

            # Show statistics if requested
            if show_stats and transcription:
                bot.print_statistics(censor_segments, transcription, badwords)

            # Dry-run mode: show what would be censored and exit
            if dry_run:
                print("\n" + "=" * 60)
                print("DRY RUN MODE - Preview of censorship")
                print("=" * 60)
                print(f"Total segments to censor: {len(censor_segments)}\n")

                if transcription:
                    # Show profane words with timestamps
                    badwords_set = set(word.lower() for word in badwords)
                    print("Profane words found:")
                    for start, end, word in transcription:
                        word_clean = word.strip().lower()
                        if word_clean in badwords_set or any(bad in word_clean for bad in badwords_set):
                            print(f"  [{start:.2f}s - {end:.2f}s] '{word}'")
                else:
                    # Show segments from subtitles
                    print("Censor segments:")
                    for i, (start, end) in enumerate(censor_segments, 1):
                        print(f"  {i}. [{start:.2f}s - {end:.2f}s]")

                print("\n" + "=" * 60)
                print("No output file created (dry-run mode)")
                print("=" * 60 + "\n")
                return True

            # Export censored subtitles if requested
            if export_srt and transcription:
                bot.export_censored_subtitles(transcription, badwords, export_srt)

            # Apply censorship to audio
            censored_audio_path = os.path.join(temp_dir, "censored_audio.wav")
            if not bot.apply_censorship(
                audio_path, censored_audio_path, censor_segments,
                mode=censor_mode, beep_file=beep_file
            ):
                logger.error("Failed to apply audio censorship")
                return False

            # Merge audio back with video
            temp_output = os.path.join(temp_dir, "temp_output.mp4")
            if not bot.merge_audio_video(
                input_path, censored_audio_path, temp_output, dual_audio=dual_audio
            ):
                logger.error("Failed to merge audio and video")
                return False

            # Embed censored subtitles if available
            if censored_subtitle_path:
                if not bot.embed_subtitles(temp_output, censored_subtitle_path, output_path):
                    logger.warning("Failed to embed subtitles, using video without subs")
                    import shutil
                    shutil.move(temp_output, output_path)
            else:
                import shutil
                shutil.move(temp_output, output_path)

            logger.info(f"✓ Successfully processed video: {output_path}")
            return True

    except Exception as e:
        logger.error(f"Failed to process video {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point for CensorBot CLI."""
    parser = argparse.ArgumentParser(
        description='CensorBot - Automatically censor profanity in video files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with auto-detection
  %(prog)s -i input.mp4 -o output.mp4

  # Use beep mode instead of mute
  %(prog)s -i input.mp4 -o output.mp4 --mode beep

  # Force CPU processing
  %(prog)s -i input.mp4 -o output.mp4 --force-cpu

  # Use custom wordlist
  %(prog)s -i input.mp4 -o output.mp4 -w custom_words.txt

  # Use external subtitles
  %(prog)s -i input.mp4 -o output.mp4 -s subtitles.srt

  # Single audio track (censored only)
  %(prog)s -i input.mp4 -o output.mp4 --single-audio

  # Dry-run mode (preview only)
  %(prog)s -i input.mp4 -o output.mp4 --dry-run

  # Export censored subtitles
  %(prog)s -i input.mp4 -o output.mp4 --export-srt censored.srt

  # Show statistics
  %(prog)s -i input.mp4 -o output.mp4 --stats

  # Use custom beep sound
  %(prog)s -i input.mp4 -o output.mp4 --mode beep --beep-file custom_beep.wav

  # Use config file
  %(prog)s --config config.yaml
        """
    )

    # Required arguments (unless using config)
    parser.add_argument('-i', '--input',
                       help='Input video file path')
    parser.add_argument('-o', '--output',
                       help='Output video file path')

    # Optional arguments
    parser.add_argument('-w', '--wordlist',
                       help='Custom badwords file (one word per line)')
    parser.add_argument('-s', '--subtitles',
                       help='External subtitle file (SRT format)')
    parser.add_argument('--model', default='base',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size (default: base)')
    parser.add_argument('--mode', default='mute',
                       choices=['mute', 'beep'],
                       help='Censoring mode: mute (silence) or beep (default: mute)')
    parser.add_argument('--padding', type=float, default=0.2,
                       help='Padding in seconds around censored segments (default: 0.2)')
    parser.add_argument('--single-audio', action='store_true',
                       help='Only keep censored audio track (remove original)')
    parser.add_argument('--no-download', action='store_true',
                       help='Disable automatic subtitle downloading')
    parser.add_argument('--gpu', action='store_true', default=True,
                       help='Enable GPU acceleration (default: True)')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU-only processing (disable GPU)')

    # New features
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview censorship without creating output file')
    parser.add_argument('--export-srt',
                       help='Export censored subtitles to SRT file')
    parser.add_argument('--beep-file',
                       help='Custom beep sound file for beep mode (WAV format)')
    parser.add_argument('--stats', action='store_true',
                       help='Display statistics about censored content')
    parser.add_argument('--config',
                       help='Load configuration from YAML file')

    args = parser.parse_args()

    # Load config file if provided
    config = {}
    if args.config:
        if not os.path.exists(args.config):
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)
        bot = CensorBot()  # Temporary instance just for loading config
        config = bot.load_config(args.config)

    # Merge config with command line args (CLI args take precedence)
    def get_arg(name, default=None):
        # Check CLI args first
        cli_value = getattr(args, name, None)
        if cli_value is not None and cli_value != default:
            return cli_value
        # Fall back to config file
        return config.get(name, default)

    # Get final values with config fallback
    input_path = get_arg('input')
    output_path = get_arg('output')
    wordlist = get_arg('wordlist')
    subtitles = get_arg('subtitles')
    model_size = get_arg('model', 'base')
    censor_mode = get_arg('mode', 'mute')
    padding = get_arg('padding', 0.2)
    single_audio = get_arg('single_audio', False)
    no_download = get_arg('no_download', False)
    use_gpu = get_arg('gpu', True)
    force_cpu = get_arg('force_cpu', False)
    dry_run = get_arg('dry_run', False)
    export_srt = get_arg('export_srt')
    beep_file = get_arg('beep_file')
    show_stats = get_arg('stats', False)

    # Validate required arguments
    if not input_path:
        logger.error("Input file path is required (use -i or --input)")
        sys.exit(1)
    if not output_path and not dry_run:
        logger.error("Output file path is required (use -o or --output), unless using --dry-run")
        sys.exit(1)

    # Validate input file exists
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Validate beep file if provided
    if beep_file and not os.path.exists(beep_file):
        logger.error(f"Beep file not found: {beep_file}")
        sys.exit(1)

    # Process video
    success = process_single_video(
        input_path=input_path,
        output_path=output_path or "dummy.mp4",  # Dummy output for dry-run
        badwords_file=wordlist,
        subtitles_file=subtitles,
        model_size=model_size,
        use_gpu=use_gpu,
        force_cpu=force_cpu,
        censor_mode=censor_mode,
        padding=padding,
        dual_audio=not single_audio,
        no_download_subs=no_download,
        dry_run=dry_run,
        export_srt=export_srt,
        beep_file=beep_file,
        show_stats=show_stats
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
