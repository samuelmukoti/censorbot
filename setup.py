"""Setup script for CensorBot package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="censorbot",
    version="0.0.5",
    author="Samuel Mukoti",
    author_email="contact@samuelmukoti.com",
    description="Automatically censor profanity in video files using AI transcription",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samuelmukoti/censorbot",
    project_urls={
        "Bug Reports": "https://github.com/samuelmukoti/censorbot/issues",
        "Source": "https://github.com/samuelmukoti/censorbot",
        "Documentation": "https://github.com/samuelmukoti/censorbot#readme",
    },
    packages=find_packages(),
    py_modules=["censor"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "faster-whisper>=1.0.0",
        "numpy>=1.24.0,<2.0.0",
        "pysrt>=1.1.2",
        "subliminal==2.1.0",
        "babelfish>=0.6.0",
        "ffmpeg-python>=0.2.0",
        "chardet>=5.0.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
        "guessit>=3.7.1",
        "click>=8.0.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "mlx": [
            "mlx>=0.4.0",
            "mlx-whisper>=0.4.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "censorbot=censor:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["badwords.txt"],
    },
    keywords="video censoring profanity subtitles whisper ai transcription",
    zip_safe=False,
)
