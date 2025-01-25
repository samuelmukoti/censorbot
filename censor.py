#!/usr/bin/env python3

import argparse
import os
import sys
import logging
from pathlib import Path
import subprocess
import json
from typing import List, Tuple, Optional, Dict
import re
import pysrt
import tempfile
from subliminal import download_best_subtitles, region, save_subtitles, scan_video
from babelfish import Language
import chardet
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import platform
import torch

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Check for Apple Silicon
IS_APPLE_SILICON = platform.processor() == 'arm' and platform.system() == 'Darwin'

# Check for CoreML support
COREML_AVAILABLE = False
if IS_APPLE_SILICON:
    try:
        import coremltools
        COREML_AVAILABLE = True
    except ImportError:
        pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default list of words to censor
DEFAULT_BADWORDS = {
    # Common profanity
    "fuck", "shit", "ass", "bitch", "damn", "hell",
    # Racial slurs
    "nigger", "nigga", "chink", "spic", "wetback", "kike", "gook",
    # Sexual terms
    "cock", "dick", "pussy", "cunt", "whore", "slut",
    # Religious profanity
    "goddamn", "jesus christ",
    # Variations and common misspellings
    "f*ck", "sh*t", "b*tch", "f**k", "s**t", "fuk", "fck", "fucker",
    # Compound words
    "motherfucker", "motherfucking", "fucking", "bullshit", "horseshit",
    "asshole", "dumbass", "jackass", "dipshit", "douchebag"
}

class CensorBot:
    def __init__(self, model_size: str = "base", use_gpu: bool = True):
        self.setup_acceleration(model_size, use_gpu)
        
        # Initialize subliminal's cache
        region.configure('dogpile.cache.dbm', arguments={'filename': 'cachefile.dbm'})
        
        # Create beep sound file
        self._create_beep_file()
        
    def setup_acceleration(self, model_size: str, use_gpu: bool):
        """Setup acceleration based on platform and available hardware."""
        self.use_gpu = use_gpu and WHISPER_AVAILABLE
        
        if not WHISPER_AVAILABLE:
            logger.warning("Faster Whisper not available. Some features may be limited.")
            return

        if IS_APPLE_SILICON and COREML_AVAILABLE:
            # Use CoreML acceleration for Apple Silicon
            logger.info("Using Apple Silicon acceleration with CoreML")
            self.model = WhisperModel(
                model_size,
                device="cpu",  # CoreML runs on CPU but uses Neural Engine
                compute_type="int8",
                cpu_threads=os.cpu_count(),
                num_workers=2
            )
        elif torch.cuda.is_available() and self.use_gpu:
            # Use CUDA acceleration for NVIDIA GPUs
            logger.info("Using NVIDIA GPU acceleration with CUDA")
            self.model = WhisperModel(
                model_size,
                device="cuda",
                compute_type="float16"
            )
        else:
            # Fall back to CPU
            logger.info("Using CPU for processing")
            self.model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=os.cpu_count()
            )

    def _create_beep_file(self):
        """Create a beep sound file using FFmpeg."""
        self.beep_file = "beep.wav"
        if not os.path.exists(self.beep_file):
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", "sine=frequency=1000:duration=0.5",
                "-af", "afade=t=in:st=0:d=0.1,afade=t=out:st=0.4:d=0.1",
                self.beep_file
            ]
            subprocess.run(cmd, check=True, capture_output=True)

    def extract_embedded_subtitles(self, video_path: str) -> Optional[str]:
        """Extract embedded subtitles from video file if they exist."""
        try:
            # Create a temporary file for the subtitles
            temp_srt = tempfile.NamedTemporaryFile(suffix='.srt', delete=False)
            temp_srt.close()
            
            # Try to extract subtitles using FFmpeg
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-map", "0:s:0",  # Select first subtitle stream
                "-f", "srt",
                temp_srt.name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if os.path.exists(temp_srt.name) and os.path.getsize(temp_srt.name) > 0:
                logger.info("Successfully extracted embedded subtitles")
                return temp_srt.name
            else:
                os.unlink(temp_srt.name)
                return None
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"No embedded subtitles found: {e}")
            if os.path.exists(temp_srt.name):
                os.unlink(temp_srt.name)
            return None

    def download_subtitles(self, video_path: str) -> Optional[str]:
        """Download subtitles from OpenSubtitles."""
        try:
            logger.info("Attempting to download subtitles...")
            video = scan_video(video_path)
            subtitles = download_best_subtitles([video], {Language('eng')})
            
            if subtitles[video]:
                # Save the first subtitle to a temporary file
                subtitle = list(subtitles[video])[0]
                temp_srt = tempfile.NamedTemporaryFile(suffix='.srt', delete=False)
                temp_srt.close()
                
                save_subtitles(video, [subtitle], encoding='utf-8')
                logger.info("Successfully downloaded subtitles")
                return temp_srt.name
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to download subtitles: {e}")
            return None

    def extract_audio(self, video_path: str, output_path: str) -> bool:
        """Extract audio from video file using FFmpeg."""
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM format
                "-ar", "16000",  # 16kHz sampling
                "-ac", "1",  # Mono
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract audio: {e}")
            return False

    def load_badwords(self, badwords_file: Optional[str] = None) -> List[str]:
        """Load bad words from file or use defaults."""
        if badwords_file:
            try:
                with open(badwords_file, 'r') as f:
                    custom_words = {word.strip().lower() for word in f.readlines() if word.strip()}
                return list(DEFAULT_BADWORDS.union(custom_words))
            except Exception as e:
                logger.warning(f"Failed to load custom badwords file: {e}. Using defaults.")
                return list(DEFAULT_BADWORDS)
        return list(DEFAULT_BADWORDS)

    def process_subtitles(self, srt_file: str, badwords: List[str]) -> List[Tuple[float, float]]:
        """Process SRT file to find timestamps of bad words."""
        try:
            # Detect the encoding of the subtitle file
            with open(srt_file, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']

            # Load subtitles with detected encoding
            subs = pysrt.open(srt_file, encoding=encoding)
            censor_segments = []
            
            for sub in subs:
                text = sub.text.lower()
                # Check if any bad word is in the subtitle text
                if any(badword in text for badword in badwords):
                    # Convert timestamp to seconds
                    start_time = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000
                    end_time = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000
                    censor_segments.append((start_time, end_time))
            
            return censor_segments
            
        except Exception as e:
            logger.error(f"Failed to process subtitles: {e}")
            return []

    def transcribe_audio(self, audio_path: str) -> List[Tuple[float, float, str]]:
        """Transcribe audio using Whisper and return word-level timestamps."""
        if not WHISPER_AVAILABLE:
            raise RuntimeError("Faster Whisper is not installed")
        
        segments, _ = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            language="en"
        )
        
        word_timestamps = []
        for segment in segments:
            for word in segment.words:
                word_timestamps.append((word.start, word.end, word.word.lower()))
        
        return word_timestamps

    def find_censor_segments(self, 
                           transcription: List[Tuple[float, float, str]], 
                           badwords: List[str],
                           padding: float = 0.2) -> List[Tuple[float, float]]:
        """Find segments to censor based on transcription and bad words."""
        censor_segments = []
        for start, end, word in transcription:
            if any(badword in word.lower() for badword in badwords):
                censor_segments.append((
                    max(0, start - padding),
                    end + padding
                ))
        return censor_segments

    def apply_censorship(self, 
                        video_path: str, 
                        censor_segments: List[Tuple[float, float]], 
                        output_path: str,
                        mode: str = "mute",
                        dual_audio: bool = True) -> bool:
        """Apply censorship to video using FFmpeg."""
        try:
            # Create complex filter for audio censoring
            filter_complex = []
            
            if mode == "beep":
                # Generate beep overlay filter
                for i, (start, end) in enumerate(censor_segments):
                    duration = end - start
                    filter_complex.extend([
                        f"[1:a]atrim=duration={duration}[beep{i}];",
                        f"[beep{i}]adelay={int(start*1000)}|{int(start*1000)}[delayed_beep{i}];"
                    ])
                
                # Mix all beeps with original audio
                beep_mix = ";".join(f"[delayed_beep{i}]" for i in range(len(censor_segments)))
                if beep_mix:
                    filter_complex.append(f"[0:a]{beep_mix}amix=inputs={len(censor_segments)+1}[censored]")
                else:
                    filter_complex.append("[0:a]acopy[censored]")
                    
            else:  # mute mode
                mute_filters = []
                for start, end in censor_segments:
                    mute_filters.append(f"volume=enable='between(t,{start},{end})':volume=0")
                
                if mute_filters:
                    filter_complex.append(f"[0:a]{','.join(mute_filters)}[censored]")
                else:
                    filter_complex.append("[0:a]acopy[censored]")
            
            filter_str = ";".join(filter_complex)
            
            # Build FFmpeg command
            cmd = ["ffmpeg", "-y", "-i", video_path]
            
            if mode == "beep":
                cmd.extend(["-i", self.beep_file])
            
            cmd.extend([
                "-filter_complex", filter_str,
                "-map", "0:v",  # Copy video stream
            ])
            
            if dual_audio:
                # Add both original and censored audio tracks
                cmd.extend([
                    "-map", "0:a",  # Original audio
                    "-map", "[censored]",  # Censored audio
                    "-c:v", "copy",  # Copy video codec
                    "-c:a:0", "aac",  # First audio stream codec
                    "-c:a:1", "aac",  # Second audio stream codec
                    "-metadata:s:a:0", "title=Original Audio",
                    "-metadata:s:a:1", "title=Censored Audio"
                ])
            else:
                # Only add censored audio track
                cmd.extend([
                    "-map", "[censored]",
                    "-c:v", "copy",
                    "-c:a", "aac"
                ])
            
            cmd.append(output_path)
            
            subprocess.run(cmd, check=True, capture_output=True)
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply censorship: {e}")
            return False

def process_video(args: Dict) -> bool:
    """Process a single video file."""
    try:
        censor_bot = CensorBot(model_size=args["model"], use_gpu=args["gpu"])
        
        # Load bad words
        badwords = censor_bot.load_badwords(args.get("wordlist"))
        
        # Try to get subtitles
        subtitle_path = None
        if args.get("subtitles"):
            subtitle_path = args["subtitles"]
        else:
            subtitle_path = censor_bot.extract_embedded_subtitles(args["input"])
            if not subtitle_path and not args.get("no_download"):
                subtitle_path = censor_bot.download_subtitles(args["input"])
        
        # Process audio/subtitles
        audio_path = None
        censor_segments = []
        
        try:
            if subtitle_path:
                logger.info(f"Processing {args['input']} using subtitles")
                censor_segments = censor_bot.process_subtitles(subtitle_path, badwords)
            else:
                logger.info(f"Processing {args['input']} using Whisper")
                audio_path = f"temp_audio_{os.path.basename(args['input'])}.wav"
                if censor_bot.extract_audio(args["input"], audio_path):
                    transcription = censor_bot.transcribe_audio(audio_path)
                    censor_segments = censor_bot.find_censor_segments(
                        transcription, badwords, args.get("padding", 0.2)
                    )
            
            # Apply censorship
            return censor_bot.apply_censorship(
                args["input"], 
                censor_segments, 
                args["output"],
                mode=args.get("mode", "mute"),
                dual_audio=args.get("dual_audio", True)
            )
            
        finally:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            if subtitle_path and subtitle_path != args.get("subtitles"):
                os.remove(subtitle_path)
                
    except Exception as e:
        logger.error(f"Failed to process {args['input']}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="CensorBot - Audio censorship tool")
    parser.add_argument("-i", "--input", required=True, help="Input video file or directory")
    parser.add_argument("-o", "--output", help="Output video file or directory")
    parser.add_argument("-w", "--wordlist", help="Optional file containing additional words to censor")
    parser.add_argument("-s", "--subtitles", help="Subtitle file (SRT format)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for transcription")
    parser.add_argument("--model", default="base", help="Whisper model size")
    parser.add_argument("--padding", type=float, default=0.2, help="Padding around censored segments")
    parser.add_argument("--no-download", action="store_true", help="Disable subtitle downloading")
    parser.add_argument("--mode", choices=["mute", "beep"], default="mute", help="Censoring mode")
    parser.add_argument("--single-audio", action="store_true", help="Only output censored audio track")
    parser.add_argument("--batch", action="store_true", help="Process all videos in input directory")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum number of parallel workers")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU processing even if GPU is available")
    
    args = parser.parse_args()
    
    # Update GPU flag based on force-cpu option
    if args.force_cpu:
        args.gpu = False
    
    # Handle batch processing
    if args.batch:
        if not os.path.isdir(args.input):
            logger.error("Input must be a directory when using --batch")
            sys.exit(1)
        if not args.output:
            args.output = args.input
        elif not os.path.isdir(args.output):
            os.makedirs(args.output)
            
        # Collect all video files
        video_files = []
        for ext in ['.mp4', '.mkv', '.avi']:
            video_files.extend(Path(args.input).glob(f'**/*{ext}'))
        
        if not video_files:
            logger.error("No video files found in input directory")
            sys.exit(1)
            
        # Prepare arguments for each video
        process_args = []
        for video_file in video_files:
            rel_path = video_file.relative_to(args.input)
            output_path = Path(args.output) / rel_path.with_name(f"{rel_path.stem}_censored{rel_path.suffix}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            process_args.append({
                "input": str(video_file),
                "output": str(output_path),
                "wordlist": args.wordlist,
                "subtitles": args.subtitles,
                "gpu": args.gpu,
                "model": args.model,
                "padding": args.padding,
                "no_download": args.no_download,
                "mode": args.mode,
                "dual_audio": not args.single_audio
            })
        
        # Process videos in parallel
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            results = list(tqdm(
                executor.map(process_video, process_args),
                total=len(process_args),
                desc="Processing videos"
            ))
        
        # Report results
        success_count = sum(1 for r in results if r)
        logger.info(f"Successfully processed {success_count} out of {len(video_files)} videos")
        
    else:
        # Single file processing
        if not args.output:
            input_path = Path(args.input)
            args.output = str(input_path.parent / f"{input_path.stem}_censored{input_path.suffix}")
        
        process_args = {
            "input": args.input,
            "output": args.output,
            "wordlist": args.wordlist,
            "subtitles": args.subtitles,
            "gpu": args.gpu,
            "model": args.model,
            "padding": args.padding,
            "no_download": args.no_download,
            "mode": args.mode,
            "dual_audio": not args.single_audio
        }
        
        if process_video(process_args):
            logger.info(f"Successfully created censored video: {args.output}")
        else:
            logger.error("Failed to create censored video")
            sys.exit(1)

if __name__ == "__main__":
    main() 