#!/usr/bin/env python3

import pytest
import os
import sys
import tempfile
import subprocess
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from pathlib import Path

# Mock missing dependencies before importing censor module
sys.modules['pysrt'] = MagicMock()
sys.modules['subliminal'] = MagicMock()
sys.modules['faster_whisper'] = MagicMock()
sys.modules['torch'] = MagicMock()

# Import the module under test
import censor
from censor import CensorBot, process_video, main, DEFAULT_BADWORDS


# Fixture to create CensorBot with all dependencies mocked
@pytest.fixture
def mock_censorbot():
    """Create a mocked CensorBot instance."""
    with patch('subprocess.run'), \
         patch('os.path.exists', return_value=False), \
         patch('censor.WHISPER_AVAILABLE', False):
        bot = CensorBot()
        yield bot


class TestCensorBotInit:
    """Test CensorBot initialization and setup."""

    @patch('censor.WHISPER_AVAILABLE', True)
    @patch('censor.IS_APPLE_SILICON', False)
    @patch('censor.torch.cuda.is_available', return_value=False)
    @patch('censor.WhisperModel')
    @patch('subprocess.run')
    @patch('os.path.exists', return_value=False)
    def test_init_cpu_only(self, mock_exists, mock_run, mock_whisper, mock_cuda):
        """Test initialization with CPU only."""
        bot = CensorBot(model_size="base", use_gpu=False)
        mock_whisper.assert_called_once()
        assert bot.use_gpu is False

    @patch('censor.WHISPER_AVAILABLE', True)
    @patch('censor.IS_APPLE_SILICON', False)
    @patch('censor.torch.cuda.is_available', return_value=True)
    @patch('censor.WhisperModel')
    @patch('subprocess.run')
    @patch('os.path.exists', return_value=False)
    def test_init_gpu_cuda(self, mock_exists, mock_run, mock_whisper, mock_cuda):
        """Test initialization with CUDA GPU."""
        bot = CensorBot(model_size="base", use_gpu=True)
        mock_whisper.assert_called_once_with(
            "base",
            device="cuda",
            compute_type="float16"
        )
        assert bot.use_gpu is True

    @patch('censor.WHISPER_AVAILABLE', True)
    @patch('censor.IS_APPLE_SILICON', True)
    @patch('censor.COREML_AVAILABLE', True)
    @patch('censor.WhisperModel')
    @patch('subprocess.run')
    @patch('os.path.exists', return_value=False)
    @patch('os.cpu_count', return_value=8)
    def test_init_apple_silicon(self, mock_cpu, mock_exists, mock_run, mock_whisper):
        """Test initialization on Apple Silicon."""
        bot = CensorBot(model_size="small", use_gpu=True)
        mock_whisper.assert_called_once_with(
            "small",
            device="cpu",
            compute_type="int8",
            cpu_threads=8,
            num_workers=2
        )

    @patch('censor.WHISPER_AVAILABLE', False)
    @patch('os.path.exists', return_value=False)
    @patch('subprocess.run')
    def test_init_no_whisper(self, mock_run, mock_exists):
        """Test initialization when Whisper is not available."""
        bot = CensorBot(model_size="base", use_gpu=True)
        assert bot.use_gpu is False


class TestCensorBotBeepFile:
    """Test beep file creation."""

    @patch('censor.WHISPER_AVAILABLE', False)
    @patch('os.path.exists', return_value=False)
    @patch('subprocess.run')
    def test_create_beep_file(self, mock_run, mock_exists):
        """Test beep file creation."""
        bot = CensorBot()

        expected_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "sine=frequency=1000:duration=0.5",
            "-af", "afade=t=in:st=0:d=0.1,afade=t=out:st=0.4:d=0.1",
            "beep.wav"
        ]
        mock_run.assert_called_with(expected_cmd, check=True, capture_output=True)

    @patch('censor.WHISPER_AVAILABLE', False)
    @patch('os.path.exists', return_value=True)
    @patch('subprocess.run')
    def test_skip_existing_beep_file(self, mock_run, mock_exists):
        """Test that beep file creation is skipped if file exists."""
        bot = CensorBot()
        # Should not call ffmpeg since file exists
        mock_run.assert_not_called()


class TestExtractEmbeddedSubtitles:
    """Test embedded subtitle extraction."""

    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('tempfile.NamedTemporaryFile')
    def test_extract_embedded_subtitles_success(self, mock_temp, mock_size, mock_exists, mock_run, mock_censorbot):
        """Test successful extraction of embedded subtitles."""
        mock_file = Mock()
        mock_file.name = "/tmp/test.srt"
        mock_temp.return_value = mock_file
        mock_exists.return_value = True
        mock_size.return_value = 100

        result = mock_censorbot.extract_embedded_subtitles("video.mp4")

        assert result == "/tmp/test.srt"

    @patch('subprocess.run')
    @patch('os.path.exists', return_value=False)
    @patch('tempfile.NamedTemporaryFile')
    @patch('os.unlink')
    def test_extract_embedded_subtitles_not_found(self, mock_unlink, mock_temp, mock_exists, mock_run, mock_censorbot):
        """Test extraction when no embedded subtitles exist."""
        mock_file = Mock()
        mock_file.name = "/tmp/test.srt"
        mock_temp.return_value = mock_file

        result = mock_censorbot.extract_embedded_subtitles("video.mp4")

        assert result is None
        mock_unlink.assert_called_with("/tmp/test.srt")

    @patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'cmd'))
    @patch('os.path.exists', return_value=True)
    @patch('tempfile.NamedTemporaryFile')
    @patch('os.unlink')
    def test_extract_embedded_subtitles_error(self, mock_unlink, mock_temp, mock_exists, mock_run, mock_censorbot):
        """Test extraction error handling."""
        mock_file = Mock()
        mock_file.name = "/tmp/test.srt"
        mock_temp.return_value = mock_file

        result = mock_censorbot.extract_embedded_subtitles("video.mp4")

        assert result is None
        mock_unlink.assert_called()


class TestDownloadSubtitles:
    """Test subtitle downloading."""

    @patch('censor.scan_video')
    @patch('censor.download_best_subtitles')
    @patch('censor.save_subtitles')
    @patch('tempfile.NamedTemporaryFile')
    def test_download_subtitles_success(self, mock_temp, mock_save, mock_download, mock_scan, mock_censorbot):
        """Test successful subtitle download."""
        mock_file = Mock()
        mock_file.name = "/tmp/test.srt"
        mock_temp.return_value = mock_file

        mock_video = Mock()
        mock_scan.return_value = mock_video

        mock_subtitle = Mock()
        mock_download.return_value = {mock_video: {mock_subtitle}}

        result = mock_censorbot.download_subtitles("video.mp4")

        assert result == "/tmp/test.srt"
        mock_scan.assert_called_once_with("video.mp4")
        mock_save.assert_called_once()

    @patch('censor.scan_video')
    @patch('censor.download_best_subtitles')
    def test_download_subtitles_not_found(self, mock_download, mock_scan, mock_censorbot):
        """Test when no subtitles are available."""
        mock_video = Mock()
        mock_scan.return_value = mock_video
        mock_download.return_value = {mock_video: set()}

        result = mock_censorbot.download_subtitles("video.mp4")

        assert result is None

    @patch('censor.scan_video', side_effect=Exception("Network error"))
    def test_download_subtitles_error(self, mock_scan, mock_censorbot):
        """Test subtitle download error handling."""
        result = mock_censorbot.download_subtitles("video.mp4")

        assert result is None


class TestExtractAudio:
    """Test audio extraction."""

    @patch('subprocess.run')
    def test_extract_audio_success(self, mock_run, mock_censorbot):
        """Test successful audio extraction."""
        result = mock_censorbot.extract_audio("video.mp4", "audio.wav")

        assert result is True
        expected_cmd = [
            "ffmpeg", "-y",
            "-i", "video.mp4",
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "audio.wav"
        ]
        mock_run.assert_called_once_with(expected_cmd, check=True, capture_output=True)

    @patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'cmd'))
    def test_extract_audio_failure(self, mock_run, mock_censorbot):
        """Test audio extraction failure."""
        result = mock_censorbot.extract_audio("video.mp4", "audio.wav")

        assert result is False


class TestLoadBadwords:
    """Test bad word loading."""

    def test_load_badwords_default(self, mock_censorbot):
        """Test loading default bad words."""
        words = mock_censorbot.load_badwords(None)

        assert isinstance(words, list)
        assert len(words) > 0
        assert "fuck" in words
        assert "shit" in words

    @patch('builtins.open', mock_open(read_data="custom1\ncustom2\n"))
    def test_load_badwords_custom_file(self, mock_censorbot):
        """Test loading custom bad words from file."""
        words = mock_censorbot.load_badwords("custom.txt")

        assert "custom1" in words
        assert "custom2" in words
        # Should also include defaults
        assert "fuck" in words

    @patch('builtins.open', side_effect=FileNotFoundError())
    def test_load_badwords_file_not_found(self, mock_file, mock_censorbot):
        """Test loading bad words when file doesn't exist."""
        words = mock_censorbot.load_badwords("missing.txt")

        # Should fall back to defaults
        assert isinstance(words, list)
        assert "fuck" in words


class TestProcessSubtitles:
    """Test subtitle processing for bad words."""

    @patch('builtins.open', new_callable=mock_open, read_data=b'test')
    @patch('chardet.detect', return_value={'encoding': 'utf-8'})
    def test_process_subtitles_with_badwords(self, mock_chardet, mock_file_open, mock_censorbot):
        """Test processing subtitles that contain bad words."""
        # Create mock subtitle objects
        mock_sub1 = Mock()
        mock_sub1.text = "This is a clean line"
        mock_sub1.start = Mock(hours=0, minutes=0, seconds=1, milliseconds=0)
        mock_sub1.end = Mock(hours=0, minutes=0, seconds=2, milliseconds=0)

        mock_sub2 = Mock()
        mock_sub2.text = "This line has fuck in it"
        mock_sub2.start = Mock(hours=0, minutes=0, seconds=3, milliseconds=500)
        mock_sub2.end = Mock(hours=0, minutes=0, seconds=5, milliseconds=0)

        with patch('pysrt.open', return_value=[mock_sub1, mock_sub2]):
            segments = mock_censorbot.process_subtitles("test.srt", ["fuck", "shit"])

        assert len(segments) == 1
        assert segments[0] == (3.5, 5.0)

    @patch('builtins.open', new_callable=mock_open, read_data=b'test')
    @patch('chardet.detect', return_value={'encoding': 'utf-8'})
    def test_process_subtitles_no_badwords(self, mock_chardet, mock_file_open, mock_censorbot):
        """Test processing subtitles with no bad words."""
        mock_sub = Mock()
        mock_sub.text = "This is completely clean"
        mock_sub.start = Mock(hours=0, minutes=0, seconds=1, milliseconds=0)
        mock_sub.end = Mock(hours=0, minutes=0, seconds=2, milliseconds=0)

        with patch('pysrt.open', return_value=[mock_sub]):
            segments = mock_censorbot.process_subtitles("test.srt", ["fuck", "shit"])

        assert len(segments) == 0

    @patch('builtins.open', side_effect=Exception("Read error"))
    def test_process_subtitles_error(self, mock_file, mock_censorbot):
        """Test subtitle processing error handling."""
        segments = mock_censorbot.process_subtitles("test.srt", ["fuck", "shit"])

        assert segments == []


class TestTranscribeAudio:
    """Test audio transcription."""

    @patch('censor.WHISPER_AVAILABLE', True)
    @patch('subprocess.run')
    @patch('os.path.exists', return_value=False)
    @patch('censor.WhisperModel')
    def test_transcribe_audio(self, mock_whisper_class, mock_exists, mock_run):
        """Test audio transcription with word timestamps."""
        # Setup mock model and transcription results
        mock_model = Mock()
        mock_whisper_class.return_value = mock_model

        # Create mock words with timestamps
        mock_word1 = Mock(start=0.0, end=0.5, word="Hello")
        mock_word2 = Mock(start=0.5, end=1.0, word="world")

        mock_segment = Mock()
        mock_segment.words = [mock_word1, mock_word2]

        mock_model.transcribe.return_value = ([mock_segment], None)

        bot = CensorBot()
        result = bot.transcribe_audio("audio.wav")

        assert len(result) == 2
        assert result[0] == (0.0, 0.5, "hello")
        assert result[1] == (0.5, 1.0, "world")

    def test_transcribe_audio_no_whisper(self, mock_censorbot):
        """Test transcription when Whisper is not available."""
        with pytest.raises(RuntimeError, match="Faster Whisper is not installed"):
            mock_censorbot.transcribe_audio("audio.wav")


class TestFindCensorSegments:
    """Test finding censor segments from transcription."""

    def test_find_censor_segments_with_badwords(self, mock_censorbot):
        """Test finding segments with bad words."""
        transcription = [
            (0.0, 0.5, "hello"),
            (0.5, 1.0, "fuck"),
            (1.0, 1.5, "world"),
            (1.5, 2.0, "shit"),
        ]

        segments = mock_censorbot.find_censor_segments(transcription, ["fuck", "shit"], padding=0.2)

        assert len(segments) == 2
        assert segments[0] == (0.3, 1.2)  # 0.5 - 0.2, 1.0 + 0.2
        assert segments[1] == (1.3, 2.2)  # 1.5 - 0.2, 2.0 + 0.2

    def test_find_censor_segments_no_badwords(self, mock_censorbot):
        """Test finding segments with no bad words."""
        transcription = [
            (0.0, 0.5, "hello"),
            (0.5, 1.0, "world"),
        ]

        segments = mock_censorbot.find_censor_segments(transcription, ["fuck", "shit"], padding=0.2)

        assert len(segments) == 0

    def test_find_censor_segments_zero_padding(self, mock_censorbot):
        """Test finding segments with zero padding."""
        transcription = [
            (0.5, 1.0, "fuck"),
        ]

        segments = mock_censorbot.find_censor_segments(transcription, ["fuck"], padding=0.0)

        assert len(segments) == 1
        assert segments[0] == (0.5, 1.0)

    def test_find_censor_segments_start_boundary(self, mock_censorbot):
        """Test that segments don't go below 0."""
        transcription = [
            (0.1, 0.3, "fuck"),
        ]

        segments = mock_censorbot.find_censor_segments(transcription, ["fuck"], padding=0.2)

        assert len(segments) == 1
        assert segments[0] == (0.0, 0.5)  # max(0, 0.1 - 0.2), 0.3 + 0.2


class TestApplyCensorship:
    """Test applying censorship to video."""

    @patch('subprocess.run')
    def test_apply_censorship_mute_mode(self, mock_run, mock_censorbot):
        """Test applying censorship in mute mode."""
        segments = [(1.0, 2.0), (3.0, 4.0)]
        result = mock_censorbot.apply_censorship(
            "input.mp4",
            segments,
            "output.mp4",
            mode="mute",
            dual_audio=True
        )

        assert result is True
        mock_run.assert_called_once()

        # Check that ffmpeg command was called
        cmd = mock_run.call_args[0][0]
        assert "ffmpeg" in cmd
        assert "-i" in cmd
        assert "input.mp4" in cmd
        assert "output.mp4" in cmd

    @patch('subprocess.run')
    def test_apply_censorship_beep_mode(self, mock_run, mock_censorbot):
        """Test applying censorship in beep mode."""
        mock_censorbot.beep_file = "beep.wav"

        segments = [(1.0, 2.0)]
        result = mock_censorbot.apply_censorship(
            "input.mp4",
            segments,
            "output.mp4",
            mode="beep",
            dual_audio=False
        )

        assert result is True
        mock_run.assert_called_once()

        # Check that beep file is included
        cmd = mock_run.call_args[0][0]
        assert "beep.wav" in cmd

    @patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'cmd'))
    def test_apply_censorship_failure(self, mock_run, mock_censorbot):
        """Test censorship application failure."""
        segments = [(1.0, 2.0)]
        result = mock_censorbot.apply_censorship(
            "input.mp4",
            segments,
            "output.mp4",
            mode="mute",
            dual_audio=True
        )

        assert result is False

    @patch('subprocess.run')
    def test_apply_censorship_no_segments(self, mock_run, mock_censorbot):
        """Test censorship with no segments to censor."""
        result = mock_censorbot.apply_censorship(
            "input.mp4",
            [],
            "output.mp4",
            mode="mute",
            dual_audio=True
        )

        assert result is True
        mock_run.assert_called_once()


class TestProcessVideo:
    """Test the process_video function."""

    @patch('censor.CensorBot')
    def test_process_video_with_subtitles(self, mock_bot_class):
        """Test processing video with provided subtitles."""
        mock_bot = Mock()
        mock_bot_class.return_value = mock_bot

        mock_bot.load_badwords.return_value = ["fuck", "shit"]
        mock_bot.process_subtitles.return_value = [(1.0, 2.0)]
        mock_bot.apply_censorship.return_value = True

        args = {
            "input": "video.mp4",
            "output": "output.mp4",
            "subtitles": "subs.srt",
            "wordlist": None,
            "model": "base",
            "gpu": False,
            "padding": 0.2,
            "mode": "mute",
            "dual_audio": True
        }

        result = process_video(args)

        assert result is True
        mock_bot.process_subtitles.assert_called_once_with("subs.srt", ["fuck", "shit"])
        mock_bot.apply_censorship.assert_called_once()

    @patch('censor.CensorBot')
    @patch('os.path.exists', return_value=True)
    @patch('os.remove')
    def test_process_video_with_transcription(self, mock_remove, mock_exists, mock_bot_class):
        """Test processing video with Whisper transcription."""
        mock_bot = Mock()
        mock_bot_class.return_value = mock_bot

        mock_bot.load_badwords.return_value = ["fuck", "shit"]
        mock_bot.extract_embedded_subtitles.return_value = None
        mock_bot.extract_audio.return_value = True
        mock_bot.transcribe_audio.return_value = [(0.0, 0.5, "fuck")]
        mock_bot.find_censor_segments.return_value = [(0.0, 0.7)]
        mock_bot.apply_censorship.return_value = True

        args = {
            "input": "video.mp4",
            "output": "output.mp4",
            "subtitles": None,
            "wordlist": None,
            "model": "base",
            "gpu": False,
            "no_download": True,
            "padding": 0.2,
            "mode": "mute",
            "dual_audio": True
        }

        result = process_video(args)

        assert result is True
        mock_bot.extract_audio.assert_called_once()
        mock_bot.transcribe_audio.assert_called_once()
        mock_bot.find_censor_segments.assert_called_once()

    @patch('censor.CensorBot')
    def test_process_video_error(self, mock_bot_class):
        """Test processing video with error."""
        mock_bot_class.side_effect = Exception("Test error")

        args = {
            "input": "video.mp4",
            "output": "output.mp4",
            "model": "base",
            "gpu": False
        }

        result = process_video(args)

        assert result is False


class TestMain:
    """Test the main CLI function."""

    @patch('sys.argv', ['censor.py', '-i', 'video.mp4'])
    @patch('censor.process_video', return_value=True)
    def test_main_single_file(self, mock_process):
        """Test main function with single file."""
        main()

        mock_process.assert_called_once()
        args = mock_process.call_args[0][0]
        assert args["input"] == "video.mp4"
        assert args["mode"] == "mute"
        assert args["dual_audio"] is True

    @patch('sys.argv', ['censor.py', '-i', 'video.mp4', '-o', 'out.mp4', '--mode', 'beep'])
    @patch('censor.process_video', return_value=True)
    def test_main_with_options(self, mock_process):
        """Test main function with various options."""
        main()

        args = mock_process.call_args[0][0]
        assert args["input"] == "video.mp4"
        assert args["output"] == "out.mp4"
        assert args["mode"] == "beep"

    @patch('sys.argv', ['censor.py', '-i', 'video.mp4', '--force-cpu'])
    @patch('censor.process_video', return_value=True)
    def test_main_force_cpu(self, mock_process):
        """Test main function with force-cpu flag."""
        main()

        args = mock_process.call_args[0][0]
        assert args["gpu"] is False

    @patch('sys.argv', ['censor.py', '-i', '/videos', '--batch', '--max-workers', '4'])
    @patch('os.path.isdir', return_value=True)
    @patch('pathlib.Path.glob')
    @patch('censor.ThreadPoolExecutor')
    @patch('censor.tqdm')
    def test_main_batch_processing(self, mock_tqdm, mock_executor_class, mock_glob, mock_isdir):
        """Test main function with batch processing."""
        # Setup mock files
        mock_files = [
            Path('/videos/video1.mp4'),
            Path('/videos/video2.mp4')
        ]
        mock_glob.return_value = mock_files

        # Setup mock executor
        mock_executor = Mock()
        mock_executor.__enter__ = Mock(return_value=mock_executor)
        mock_executor.__exit__ = Mock(return_value=False)
        mock_executor.map.return_value = [True, True]
        mock_executor_class.return_value = mock_executor

        # Setup mock tqdm
        mock_tqdm.return_value = [True, True]

        main()

        mock_executor_class.assert_called_once_with(max_workers=4)

    @patch('sys.argv', ['censor.py', '-i', '/videos', '--batch'])
    @patch('os.path.isdir', return_value=False)
    def test_main_batch_not_directory(self, mock_isdir):
        """Test main function batch mode with non-directory input."""
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @patch('sys.argv', ['censor.py', '-i', '/videos', '--batch'])
    @patch('os.path.isdir', return_value=True)
    @patch('pathlib.Path.glob', return_value=[])
    def test_main_batch_no_files(self, mock_glob, mock_isdir):
        """Test main function batch mode with no video files found."""
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @patch('sys.argv', ['censor.py', '-i', 'video.mp4'])
    @patch('censor.process_video', return_value=False)
    def test_main_process_failure(self, mock_process):
        """Test main function when processing fails."""
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


class TestDefaultBadwords:
    """Test the default badwords list."""

    def test_default_badwords_not_empty(self):
        """Test that default badwords list is not empty."""
        assert len(DEFAULT_BADWORDS) > 0

    def test_default_badwords_contains_common_profanity(self):
        """Test that default badwords contains common profanity."""
        assert "fuck" in DEFAULT_BADWORDS
        assert "shit" in DEFAULT_BADWORDS
        assert "ass" in DEFAULT_BADWORDS

    def test_default_badwords_all_lowercase(self):
        """Test that all default badwords are lowercase."""
        for word in DEFAULT_BADWORDS:
            assert word == word.lower()


class TestModuleConstants:
    """Test module-level constants and flags."""

    def test_whisper_available_flag(self):
        """Test that WHISPER_AVAILABLE flag is set."""
        assert isinstance(censor.WHISPER_AVAILABLE, bool)

    def test_is_apple_silicon_flag(self):
        """Test that IS_APPLE_SILICON flag is set."""
        assert isinstance(censor.IS_APPLE_SILICON, bool)

    def test_coreml_available_flag(self):
        """Test that COREML_AVAILABLE flag is set."""
        assert isinstance(censor.COREML_AVAILABLE, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=censor", "--cov-report=term-missing"])
