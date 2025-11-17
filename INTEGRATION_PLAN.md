# CensorBot Integration Plan for Jellyfin & Kodi

Comprehensive guide for integrating CensorBot profanity censoring functionality into Jellyfin and Kodi media platforms.

## Executive Summary

**Goal**: Enable automatic profanity censoring during media playback in Jellyfin and Kodi

**Approach**: Three integration strategies based on platform architecture and use cases

**Timeline**: 2-6 months depending on approach complexity

---

## Platform Architecture Overview

### Jellyfin
- **Language**: C# / .NET 8.0
- **Plugin System**: Interface-based dependency injection
- **Integration Points**:
  - Media processing pipelines
  - Transcoding hooks
  - Scheduled tasks
  - REST API endpoints
- **Deployment**: DLL plugins in `/var/lib/jellyfin/plugins/`

### Kodi
- **Language**: Python + XML
- **Addon System**: Python-based addon framework
- **Integration Points**:
  - Player event hooks
  - Context menu actions
  - Service addons (background processing)
  - Script addons (on-demand execution)
- **Deployment**: ZIP addons in userdata/addons/

---

## Integration Strategy Comparison

| Approach | Jellyfin | Kodi | Complexity | Real-time |
|----------|----------|------|------------|-----------|
| **1. Pre-Processing** | ⭐⭐⭐ Best | ⭐⭐⭐ Best | Low | No |
| **2. Real-time Streaming** | ⭐⭐ Good | ⭐ Limited | High | Yes |
| **3. Hybrid On-Demand** | ⭐⭐⭐ Best | ⭐⭐ Good | Medium | Cached |

---

## Strategy 1: Pre-Processing Approach (RECOMMENDED)

### Overview
Process videos in advance, store censored versions, serve on playback.

### Architecture

```
┌─────────────────┐
│  Media Library  │
│   (Original)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   CensorBot     │◄──── Scheduled Task (Jellyfin)
│   Processing    │      Service Addon (Kodi)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Censored Files  │
│  (Dual Audio)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Playback      │
│  (User Choice)  │
└─────────────────┘
```

### Jellyfin Implementation

**Plugin Components**:

1. **ScheduledTask Implementation** (`ICensoringTask`)
   - Scans library for new/unwatched content
   - Triggers CensorBot processing via Python subprocess
   - Updates metadata with censored track info

2. **Configuration Page** (`IPluginConfigurationPage`)
   - Enable/disable auto-processing
   - Wordlist management
   - Processing schedule (nightly, on-add, manual)
   - Model selection (tiny/base/small for speed)

3. **API Controller** (`ControllerBase`)
   - Manual trigger endpoint: `POST /CensorBot/Process/{itemId}`
   - Status check: `GET /CensorBot/Status/{itemId}`
   - Batch processing: `POST /CensorBot/ProcessLibrary`

**Technical Details**:

```csharp
// Plugin entry point
public class CensorBotPlugin : BasePlugin<PluginConfiguration>
{
    public override string Name => "CensorBot";
    public override Guid Id => Guid.Parse("12345678-1234-1234-1234-123456789abc");

    public CensorBotPlugin(IApplicationPaths applicationPaths, IXmlSerializer xmlSerializer)
        : base(applicationPaths, xmlSerializer)
    {
    }
}

// Scheduled task for processing
public class CensoringTask : IScheduledTask
{
    public string Name => "Process Videos with CensorBot";
    public string Key => "CensorBotProcessing";

    public async Task ExecuteAsync(IProgress<double> progress, CancellationToken cancellationToken)
    {
        // 1. Get unprocessed videos from library
        // 2. Call CensorBot Python CLI via Process.Start()
        // 3. Update media items with dual audio track info
        // 4. Mark as processed in plugin database
    }

    public IEnumerable<TaskTriggerInfo> GetDefaultTriggers()
    {
        return new[]
        {
            new TaskTriggerInfo { Type = TaskTriggerInfo.TriggerDaily, TimeOfDayTicks = TimeSpan.FromHours(2).Ticks }
        };
    }
}
```

**CensorBot Invocation**:

```csharp
var process = new Process
{
    StartInfo = new ProcessStartInfo
    {
        FileName = "pipx",
        Arguments = $"run censorbot -i \"{inputPath}\" -o \"{outputPath}\" --mode mute",
        RedirectStandardOutput = true,
        UseShellExecute = false,
        CreateNoWindow = true
    }
};

process.Start();
await process.WaitForExitAsync(cancellationToken);
```

**Advantages**:
- ✅ No playback latency
- ✅ Leverages existing dual-audio track feature
- ✅ Works with all Jellyfin clients
- ✅ Simple architecture, easy to maintain

**Disadvantages**:
- ❌ Storage overhead (censored copies)
- ❌ Processing time for large libraries
- ❌ Not real-time (scheduled)

---

### Kodi Implementation

**Addon Components**:

1. **Service Addon** (`service.censorbot`)
   - Background monitor for library updates
   - Automatic processing on media add
   - Settings integration

2. **Context Menu Action**
   - Right-click "Censor This Video"
   - Manual processing trigger
   - Progress notification

3. **Settings Interface** (`settings.xml`)
   - Enable/disable auto-processing
   - Wordlist configuration
   - Processing options

**Technical Details**:

```python
# addon.py - Main service
import xbmc
import xbmcaddon
import subprocess
from pathlib import Path

class CensorBotService(xbmc.Monitor):
    def __init__(self):
        super().__init__()
        self.addon = xbmcaddon.Addon()

    def onScanFinished(self, library):
        """Triggered when library scan completes"""
        if self.addon.getSettingBool('auto_process'):
            self.process_new_videos()

    def process_video(self, video_path):
        """Process single video with CensorBot"""
        output_path = self.get_censored_path(video_path)

        # Call CensorBot via subprocess
        result = subprocess.run([
            'pipx', 'run', 'censorbot',
            '-i', video_path,
            '-o', output_path,
            '--mode', 'mute'
        ], capture_output=True)

        if result.returncode == 0:
            # Update Kodi database with censored version
            self.update_library_entry(video_path, output_path)
            xbmc.executebuiltin(f'Notification(CensorBot, Processed: {Path(video_path).name})')

    def get_censored_path(self, original_path):
        """Generate censored file path"""
        path = Path(original_path)
        return str(path.parent / f"{path.stem}_CENSORED{path.suffix}")

if __name__ == '__main__':
    service = CensorBotService()
    service.waitForAbort()  # Keep service running
```

```xml
<!-- addon.xml -->
<?xml version="1.0" encoding="UTF-8"?>
<addon id="service.censorbot" name="CensorBot" version="1.0.0" provider-name="Your Name">
    <requires>
        <import addon="xbmc.python" version="3.0.0"/>
    </requires>
    <extension point="xbmc.service" library="addon.py"/>
    <extension point="xbmc.addon.metadata">
        <summary lang="en_GB">Automatic profanity censoring for videos</summary>
        <description lang="en_GB">Automatically detects and censors profanity in video files using AI transcription</description>
        <platform>all</platform>
        <license>GPL-3.0</license>
    </extension>
</addon>
```

```xml
<!-- settings.xml -->
<settings>
    <category label="General">
        <setting id="auto_process" type="bool" label="Auto-process new videos" default="false"/>
        <setting id="wordlist_path" type="text" label="Custom wordlist path" default=""/>
        <setting id="model_size" type="select" label="Whisper model size" values="tiny|base|small" default="base"/>
        <setting id="mode" type="select" label="Censoring mode" values="mute|beep" default="mute"/>
    </category>
</settings>
```

**Context Menu** (`context.py`):

```python
import sys
import xbmc
import xbmcaddon
from addon import CensorBotService

# Get selected item
item_path = sys.listitem.getPath()

# Process video
service = CensorBotService()
service.process_video(item_path)
```

**Advantages**:
- ✅ Native Kodi integration
- ✅ User-friendly context menu
- ✅ Background processing
- ✅ Python-based (matches CensorBot)

**Disadvantages**:
- ❌ Kodi must have Python 3.8+
- ❌ FFmpeg must be installed separately
- ❌ Storage overhead

---

## Strategy 2: Real-Time Streaming Approach

### Overview
Censor audio during playback/transcoding without creating new files.

### Jellyfin Implementation

**Approach**: Hook into transcoding pipeline

**Technical Challenges**:
1. Jellyfin transcoding uses FFmpeg directly - need to inject custom filters
2. Requires profanity timestamps cached/pre-computed
3. Complex state management for streaming sessions

**Architecture**:

```
Playback Request
    ↓
Check Profanity Cache
    ↓
├─ Cached? → Apply mute filters to transcode
│                ↓
│            Real-time censored stream
│
└─ Not Cached? → Process video first
                      ↓
                  Cache timestamps
                      ↓
                  Retry with cache
```

**Implementation Complexity**: HIGH (requires deep Jellyfin internals knowledge)

**Recommendation**: ❌ Not recommended - pre-processing is simpler and more reliable

---

### Kodi Implementation

**Approach**: Player hooks with audio filtering

**Technical Challenges**:
1. Kodi Python API doesn't provide low-level audio stream access
2. Would require binary addon (C++) for real-time audio manipulation
3. Cross-platform audio filtering complexity

**Recommendation**: ❌ Not feasible with Python addons alone

---

## Strategy 3: Hybrid On-Demand Approach (RECOMMENDED FOR LARGE LIBRARIES)

### Overview
Generate censored versions on-demand (first playback), cache for subsequent plays.

### Jellyfin Implementation

**Components**:

1. **Playback Hook** (`IPlaybackReporter`)
   - Detect when user plays uncensored content
   - Check if censored version exists
   - Trigger processing if missing

2. **Background Queue** (`IHostedService`)
   - Queue processing requests
   - Process asynchronously
   - Notify user when ready

3. **Caching Strategy**
   - LRU cache for censored files
   - Configurable cache size limit
   - Auto-cleanup of old censored versions

**User Experience**:

```
User clicks play
    ↓
Censored version exists?
    ↓
├─ Yes → Play immediately
│
└─ No → Show notification "Processing for family-friendly playback..."
            ↓
        Play original (OR wait for censored if configured)
            ↓
        Background processing
            ↓
        Next playback: Censored version ready
```

**Advantages**:
- ✅ No upfront processing time
- ✅ Only processes watched content
- ✅ Storage-efficient with LRU cache

**Disadvantages**:
- ❌ First playback delay
- ❌ More complex state management

---

## Technical Integration Details

### Calling CensorBot from C# (Jellyfin)

**Option 1: Python Subprocess**

```csharp
public async Task<bool> ProcessVideoAsync(string inputPath, string outputPath, CancellationToken cancellationToken)
{
    var startInfo = new ProcessStartInfo
    {
        FileName = "pipx",
        Arguments = $"run censorbot -i \"{inputPath}\" -o \"{outputPath}\" --mode mute --single-audio",
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false,
        CreateNoWindow = true
    };

    using var process = new Process { StartInfo = startInfo };

    var outputBuilder = new StringBuilder();
    process.OutputDataReceived += (sender, e) => {
        if (e.Data != null)
        {
            outputBuilder.AppendLine(e.Data);
            // Parse progress from output for UI updates
            LogProgress(e.Data);
        }
    };

    process.Start();
    process.BeginOutputReadLine();

    await process.WaitForExitAsync(cancellationToken);

    return process.ExitCode == 0;
}
```

**Option 2: REST API Wrapper** (Future Enhancement)

Create a lightweight REST API around CensorBot:

```python
# censorbot_api.py
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import censor

app = FastAPI()

class ProcessRequest(BaseModel):
    input_path: str
    output_path: str
    mode: str = "mute"

@app.post("/process")
async def process_video(request: ProcessRequest, background_tasks: BackgroundTasks):
    # Queue processing
    background_tasks.add_task(censor.main, [
        '-i', request.input_path,
        '-o', request.output_path,
        '--mode', request.mode
    ])
    return {"status": "queued"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    # Return processing status
    return {"status": "processing", "progress": 45}
```

Then call from C#:

```csharp
using var client = new HttpClient();
var request = new { input_path = inputPath, output_path = outputPath, mode = "mute" };
var response = await client.PostAsJsonAsync("http://localhost:8000/process", request);
```

---

### Calling CensorBot from Python (Kodi)

**Option 1: Direct Subprocess** (Current Approach)

```python
import subprocess

def process_video(input_path, output_path, mode='mute'):
    """Process video with CensorBot"""
    result = subprocess.run([
        'pipx', 'run', 'censorbot',
        '-i', input_path,
        '-o', output_path,
        '--mode', mode,
        '--single-audio'
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(f"CensorBot failed: {result.stderr}")

    return True
```

**Option 2: Import as Module** (Future Enhancement)

If CensorBot is refactored as a library:

```python
from censorbot import CensorBot

bot = CensorBot(
    input_file=input_path,
    output_file=output_path,
    mode='mute',
    model='base'
)

# Process with progress callback
def on_progress(stage, percent):
    xbmc.executebuiltin(f'SetProperty(CensorBot.Progress,{percent},10000)')

bot.process(progress_callback=on_progress)
```

---

## Deployment Strategies

### Jellyfin Plugin Packaging

**Directory Structure**:

```
Jellyfin.Plugin.CensorBot/
├── Jellyfin.Plugin.CensorBot.csproj
├── Plugin.cs
├── Configuration/
│   ├── PluginConfiguration.cs
│   └── configPage.html
├── ScheduledTasks/
│   └── CensoringTask.cs
├── Api/
│   └── CensorBotController.cs
└── build/
    └── Jellyfin.Plugin.CensorBot.dll
```

**Build & Deploy**:

```bash
# Build plugin
dotnet build --configuration Release

# Create plugin folder
mkdir -p /var/lib/jellyfin/plugins/CensorBot_1.0.0.0/

# Copy DLL
cp bin/Release/net8.0/Jellyfin.Plugin.CensorBot.dll /var/lib/jellyfin/plugins/CensorBot_1.0.0.0/

# Restart Jellyfin
systemctl restart jellyfin
```

**Distribution**:
- Create GitHub releases with DLL
- Submit to Jellyfin plugin repository
- Provide installation instructions

---

### Kodi Addon Packaging

**Directory Structure**:

```
service.censorbot/
├── addon.xml
├── addon.py
├── context.py
├── settings.xml
├── resources/
│   ├── language/
│   │   └── resource.language.en_gb/
│   │       └── strings.po
│   └── lib/
│       └── censorbot_wrapper.py
└── icon.png
```

**Build & Deploy**:

```bash
# Create ZIP
zip -r service.censorbot-1.0.0.zip service.censorbot/

# Install via Kodi UI:
# Settings → Add-ons → Install from zip file
```

**Distribution**:
- GitHub releases with ZIP
- Submit to Kodi addon repository
- Provide installation guide

---

## Implementation Roadmap

### Phase 1: Jellyfin Plugin (Weeks 1-6)

**Week 1-2: Setup & Architecture**
- [ ] Clone Jellyfin plugin template
- [ ] Design plugin architecture
- [ ] Setup development environment
- [ ] Create basic plugin structure

**Week 3-4: Core Processing**
- [ ] Implement ScheduledTask for library scanning
- [ ] Add CensorBot subprocess invocation
- [ ] Create configuration page
- [ ] Add database for tracking processed files

**Week 5: API & UI**
- [ ] Implement REST API controller
- [ ] Add manual processing endpoints
- [ ] Create admin dashboard page
- [ ] Add progress notifications

**Week 6: Testing & Polish**
- [ ] Test with various video formats
- [ ] Error handling & logging
- [ ] Performance optimization
- [ ] Documentation

**Deliverables**:
- Jellyfin plugin DLL
- Installation guide
- User documentation
- GitHub repository

---

### Phase 2: Kodi Addon (Weeks 7-10)

**Week 7: Setup & Architecture**
- [ ] Create addon structure
- [ ] Setup development environment
- [ ] Design service architecture
- [ ] Create settings interface

**Week 8: Service Implementation**
- [ ] Implement background service
- [ ] Add library monitor hooks
- [ ] Create processing queue
- [ ] Add progress notifications

**Week 9: Context Menu & UI**
- [ ] Implement context menu action
- [ ] Add settings page
- [ ] Create status indicators
- [ ] User notifications

**Week 10: Testing & Polish**
- [ ] Test on multiple Kodi platforms
- [ ] Error handling & logging
- [ ] Performance optimization
- [ ] Documentation

**Deliverables**:
- Kodi addon ZIP
- Installation guide
- User documentation
- GitHub repository

---

### Phase 3: Enhancement & Optimization (Weeks 11-12)

**Week 11: Advanced Features**
- [ ] LRU cache implementation
- [ ] Batch processing optimization
- [ ] Multi-language support
- [ ] Custom wordlist UI

**Week 12: Polish & Release**
- [ ] Final testing
- [ ] Create demo videos
- [ ] Submit to plugin repositories
- [ ] Marketing & announcement

---

## Technical Challenges & Solutions

### Challenge 1: Performance on Large Libraries

**Problem**: Processing 1000+ videos takes hours/days

**Solutions**:
1. **Prioritize Recently Added**: Process new content first
2. **User Watchlist**: Only process user's watchlist
3. **Parallel Processing**: Multiple CensorBot instances
4. **GPU Acceleration**: Leverage CUDA for faster transcription
5. **Incremental Processing**: Process in batches overnight

**Implementation**:

```csharp
// Priority-based processing queue
public class ProcessingQueue
{
    private readonly PriorityQueue<MediaItem, int> _queue = new();

    public void Enqueue(MediaItem item)
    {
        int priority = CalculatePriority(item);
        _queue.Enqueue(item, priority);
    }

    private int CalculatePriority(MediaItem item)
    {
        int priority = 0;

        // Recently added = high priority
        if (item.DateAdded > DateTime.Now.AddDays(-7))
            priority += 100;

        // In user's watchlist = high priority
        if (item.IsInWatchlist)
            priority += 50;

        // Popular items = higher priority
        priority += item.PlayCount * 10;

        return priority;
    }
}
```

---

### Challenge 2: Storage Management

**Problem**: Censored files double storage requirements

**Solutions**:
1. **LRU Cache**: Keep only recently played censored versions
2. **User Preference**: Only cache if user has "family mode" enabled
3. **Compression**: Use efficient codecs (H.265)
4. **Single Audio Track**: Use `--single-audio` flag
5. **Configurable Retention**: Delete after N days/plays

**Implementation**:

```csharp
public class CensoredFileCache
{
    private readonly Dictionary<string, CachedFile> _cache = new();
    private readonly long _maxCacheSizeBytes;
    private long _currentCacheSize;

    public async Task AddToCache(string originalPath, string censoredPath)
    {
        var fileInfo = new FileInfo(censoredPath);

        // Evict old files if cache full
        while (_currentCacheSize + fileInfo.Length > _maxCacheSizeBytes)
        {
            EvictLeastRecentlyUsed();
        }

        _cache[originalPath] = new CachedFile
        {
            Path = censoredPath,
            LastAccessed = DateTime.Now,
            Size = fileInfo.Length
        };

        _currentCacheSize += fileInfo.Length;
    }

    private void EvictLeastRecentlyUsed()
    {
        var oldest = _cache.Values.OrderBy(f => f.LastAccessed).First();
        File.Delete(oldest.Path);
        _currentCacheSize -= oldest.Size;
        _cache.Remove(_cache.First(kvp => kvp.Value == oldest).Key);
    }
}
```

---

### Challenge 3: Cross-Platform Compatibility

**Problem**: Different paths, permissions, FFmpeg locations

**Solutions**:
1. **Docker Container**: Run CensorBot in container
2. **Path Mapping**: Map Jellyfin/Kodi paths to container
3. **Permissions**: Run container with same UID as media server
4. **Auto-Detection**: Detect FFmpeg location automatically

**Jellyfin Docker Compose**:

```yaml
version: '3.8'
services:
  jellyfin:
    image: jellyfin/jellyfin
    volumes:
      - /media:/media
      - jellyfin_config:/config

  censorbot-worker:
    image: samuelmukoti/censorbot:latest
    volumes:
      - /media:/media  # Same mount as Jellyfin
      - censorbot_cache:/cache
    environment:
      - PROCESSING_QUEUE_PATH=/cache/queue
    user: "1000:1000"  # Match Jellyfin UID
```

---

### Challenge 4: User Experience

**Problem**: Users don't know processing is happening

**Solutions**:
1. **Progress Notifications**: Real-time progress updates
2. **Dashboard Widget**: Show processing queue status
3. **Email Notifications**: Alert when batch completes
4. **Auto-Play Original**: Play original while processing happens

**Jellyfin Notification**:

```csharp
public class CensorBotNotifier
{
    private readonly INotificationManager _notificationManager;

    public async Task NotifyProcessingComplete(string title)
    {
        var notification = new NotificationRequest
        {
            Name = "CensorBot Processing Complete",
            Description = $"'{title}' is now available in family-friendly mode",
            NotificationType = NotificationType.TaskFailed,
            Level = NotificationLevel.Normal,
            Url = $"/web/index.html#!/item?id={itemId}"
        };

        await _notificationManager.SendNotification(notification, CancellationToken.None);
    }
}
```

---

## Recommended Implementation Plan

### For Small/Medium Libraries (<500 videos)

**Approach**: Strategy 1 (Pre-Processing) with scheduled tasks

**Steps**:
1. Implement Jellyfin plugin with nightly scheduled task
2. Process entire library over 1-2 weeks
3. Auto-process new additions on library scan
4. Serve dual-audio files to all clients

**Timeline**: 6-8 weeks

---

### For Large Libraries (>500 videos)

**Approach**: Strategy 3 (Hybrid On-Demand) with prioritization

**Steps**:
1. Implement on-demand processing trigger
2. Priority-based queue (watchlist, recently added, popular)
3. LRU cache with configurable size limit
4. Background processing during off-peak hours

**Timeline**: 10-12 weeks

---

## Future Enhancements

### 1. CensorBot Library Mode

Refactor CensorBot to be importable as a Python library:

```python
from censorbot import CensorBot

bot = CensorBot()
result = bot.process(
    input_path='movie.mp4',
    output_path='movie_censored.mp4',
    mode='mute',
    progress_callback=lambda p: print(f"Progress: {p}%")
)
```

**Benefits**:
- Direct import in Python-based addons (Kodi)
- Better error handling and logging integration
- Progress callbacks for UI updates

---

### 2. Timestamp-Only Mode

Generate and cache only profanity timestamps, apply censoring during transcoding:

```json
{
  "movie.mp4": {
    "profanity_segments": [
      {"start": 125.3, "end": 126.1, "word": "f***"},
      {"start": 340.5, "end": 341.2, "word": "s***"}
    ],
    "processed_date": "2025-11-14",
    "model": "base"
  }
}
```

**Jellyfin Integration**:
- Store timestamps in plugin database
- Inject volume filters into FFmpeg transcode command
- No storage overhead (just metadata)

---

### 3. User Profiles & Preferences

Different censoring levels per user profile:

```json
{
  "profiles": {
    "kids": {"wordlist": "strict", "mode": "beep"},
    "teens": {"wordlist": "moderate", "mode": "mute"},
    "adults": {"enabled": false}
  }
}
```

---

### 4. Cloud Processing Service

Offer cloud-based processing for users without powerful hardware:

```
User uploads video → Cloud processes → Downloads censored version
```

**Monetization**: Subscription for cloud processing credits

---

## Cost Analysis

### Development Costs

| Phase | Estimated Hours | Cost @ $100/hr |
|-------|----------------|----------------|
| Jellyfin Plugin | 120-160 hrs | $12,000-$16,000 |
| Kodi Addon | 80-120 hrs | $8,000-$12,000 |
| Testing & QA | 40-60 hrs | $4,000-$6,000 |
| Documentation | 20-30 hrs | $2,000-$3,000 |
| **Total** | **260-370 hrs** | **$26,000-$37,000** |

### Operational Costs (Cloud Service Option)

| Resource | Monthly Cost |
|----------|--------------|
| Processing Server (GPU) | $300-$500 |
| Storage (1TB) | $20-$50 |
| Bandwidth (10TB) | $100-$200 |
| **Total** | **$420-$750/mo** |

---

## Success Metrics

### User Adoption
- Plugin installs: Target 1000+ in first 6 months
- Active users: Target 500+ monthly active users
- Processing volume: Target 10,000+ videos processed

### Performance
- Processing speed: <1x video length (1hr video in <1hr)
- Success rate: >95% successful processing
- User satisfaction: >4.5/5 average rating

### Technical
- Crash rate: <0.1% of processing jobs
- Storage efficiency: <1.5x original file size
- Cache hit rate: >70% for on-demand approach

---

## Conclusion

**Recommended Immediate Next Steps**:

1. **Proof of Concept** (Week 1-2)
   - Create minimal Jellyfin plugin that calls `pipx run censorbot`
   - Test on 5-10 videos
   - Validate dual-audio playback

2. **User Feedback** (Week 3)
   - Share PoC with beta testers
   - Gather requirements and preferences
   - Refine implementation approach

3. **Full Development** (Week 4-12)
   - Implement chosen strategy
   - Build Jellyfin plugin
   - Build Kodi addon
   - Testing and refinement

**Final Recommendation**: Start with **Jellyfin Strategy 1 (Pre-Processing)** as it's the most straightforward, reliable, and provides the best user experience. Once proven successful, expand to Kodi and explore advanced features like on-demand processing and cloud services.

---

## References

- [Jellyfin Plugin Template](https://github.com/jellyfin/jellyfin-plugin-template)
- [Jellyfin Plugin Documentation](https://jellyfin.org/docs/general/server/plugins/)
- [Kodi Addon Development](https://kodi.wiki/view/Add-on_development)
- [Kodi Python API](https://codedocs.xyz/xbmc/xbmc/)
- [CensorBot GitHub](https://github.com/samuelmukoti/censorbot)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-14
**Author**: Claude Code + Samuel Mukoti
