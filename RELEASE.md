# Release Management Guide

This document describes how to create and publish releases for CensorBot.

## Release Process

### Prerequisites

1. **PyPI Trusted Publishing** (Recommended - No tokens needed!)
   - Go to https://pypi.org/manage/account/publishing/
   - Add GitHub repository: `samuelmukoti/censorbot`
   - Workflow: `release.yml`
   - Environment: `release` (optional)

2. **Docker Hub Secrets**
   - Go to GitHub repository → Settings → Secrets and variables → Actions
   - Add secrets:
     - `DOCKERHUB_USERNAME`: Your Docker Hub username
     - `DOCKERHUB_TOKEN`: Docker Hub access token (create at https://hub.docker.com/settings/security)

### Option 1: Automated Release (Recommended)

This is the recommended approach - fully automated publishing to PyPI and Docker Hub.

#### Step 1: Update Version

Update version in these files:
- `setup.py` → `version="2.0.0"`
- `pyproject.toml` → `version = "2.0.0"`

#### Step 2: Commit and Push

```bash
git add setup.py pyproject.toml
git commit -m "Bump version to 2.0.0"
git push origin main
```

#### Step 3: Create GitHub Release

```bash
# Create and push tag
git tag -a v2.0.0 -m "Release v2.0.0: Major rewrite with MLX acceleration and new features"
git push origin v2.0.0

# Or create release via GitHub CLI
gh release create v2.0.0 \
  --title "v2.0.0: Complete Rewrite with MLX Acceleration" \
  --notes "$(cat <<'EOF'
## What's New in v2.0.0

### Major Features
- ✅ Complete rewrite with simplified FFmpeg filtering
- ✅ Real MLX hardware acceleration for Apple Silicon
- ✅ Automatic fallback mechanism (MLX → CPU)
- ✅ PyPI distribution: `pip install censorbot`

### New Productivity Features
- 🎯 Dry-run mode: Preview before processing
- 📄 Export censored subtitles (SRT)
- 📊 Word statistics reports
- 🔊 Custom beep sounds
- ⚙️ YAML configuration files
- 📊 Real-time progress bars

### Installation
**Via Pip:**
\`\`\`bash
pip install censorbot
\`\`\`

**Via Docker:**
\`\`\`bash
docker pull samuelmukoti/censorbot:2.0.0
\`\`\`

See [README.md](https://github.com/samuelmukoti/censorbot#readme) for full documentation.
EOF
)"
```

#### Step 4: Automated Publishing

Once the release is created, GitHub Actions automatically:
1. ✅ Builds Python package
2. ✅ Publishes to PyPI (using trusted publishing - no tokens!)
3. ✅ Builds Docker image for linux/amd64 and linux/arm64
4. ✅ Pushes to Docker Hub with version tag and `latest`
5. ✅ Updates release notes with installation instructions

**Monitor Progress**:
- Go to Actions tab: https://github.com/samuelmukoti/censorbot/actions
- Watch the "Release to PyPI and Docker Hub" workflow

### Option 2: Manual Local Release

If you prefer to publish manually from your dev environment:

#### PyPI Publishing

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Check package
twine check dist/*

# Upload to TestPyPI (optional - for testing)
twine upload --repository testpypi dist/*

# Upload to PyPI (production)
twine upload dist/*
# Enter your PyPI username and password or API token
```

#### Docker Publishing

```bash
# Build for multiple platforms
docker buildx create --use
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t samuelmukoti/censorbot:2.0.0 \
  -t samuelmukoti/censorbot:latest \
  --push .
```

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking changes
- **MINOR** (0.X.0): New features, backwards compatible
- **PATCH** (0.0.X): Bug fixes, backwards compatible

### Examples:
- `2.0.0` - Major rewrite (current)
- `2.1.0` - Add new feature (e.g., multi-language support)
- `2.1.1` - Fix bug in transcription

## Release Checklist

Before creating a release:

- [ ] Update version in `setup.py`
- [ ] Update version in `pyproject.toml`
- [ ] Update `README.md` if needed
- [ ] Run tests: `pytest` (if tests exist)
- [ ] Build and test locally: `python -m build && pip install dist/*.whl`
- [ ] Test installation: `censorbot --help`
- [ ] Update CHANGELOG (if exists)
- [ ] Commit version bump
- [ ] Create release via GitHub
- [ ] Verify automated workflows succeed
- [ ] Test PyPI installation: `pip install censorbot==2.0.0`
- [ ] Test Docker image: `docker pull samuelmukoti/censorbot:2.0.0`

## Troubleshooting

### PyPI Trusted Publishing Failed

If automated PyPI publishing fails:

1. **Check PyPI Configuration**:
   - Go to https://pypi.org/manage/account/publishing/
   - Verify repository name matches exactly: `samuelmukoti/censorbot`
   - Verify workflow name: `release.yml`

2. **Manual Publish Fallback**:
   ```bash
   python -m build
   twine upload dist/*
   ```

### Docker Hub Publishing Failed

1. **Check Secrets**:
   - GitHub → Settings → Secrets
   - Verify `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` exist

2. **Manual Publish Fallback**:
   ```bash
   docker login
   docker buildx build --platform linux/amd64,linux/arm64 \
     -t samuelmukoti/censorbot:2.0.0 \
     -t samuelmukoti/censorbot:latest \
     --push .
   ```

### Version Mismatch

If PyPI shows wrong version:
1. Check `setup.py` and `pyproject.toml` match
2. Clean build artifacts: `rm -rf build dist *.egg-info`
3. Rebuild: `python -m build`

## Post-Release

After successful release:

1. **Announce on Social Media**:
   - Twitter, Reddit, HackerNews, etc.
   - Include installation command and key features

2. **Update Documentation**:
   - Ensure README reflects latest features
   - Update Docker Hub description if needed

3. **Monitor Issues**:
   - Watch for bug reports from new users
   - Be ready for quick patch releases if needed

## Current Release Status

- **Latest Version**: 2.0.0
- **PyPI**: https://pypi.org/project/censorbot/
- **Docker Hub**: https://hub.docker.com/r/samuelmukoti/censorbot
- **Releases**: https://github.com/samuelmukoti/censorbot/releases
