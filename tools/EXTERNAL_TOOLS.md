# External command-line tools

Platform-specific executables are not stored in the source repository.

`make_preview_video.py` requires `ffmpeg` on `PATH`. Install a trusted, pinned
FFmpeg distribution through the deployment operating system or package manager,
and record its version in the deployment manifest.

The former `gopro2json.exe` is not used by the active BearVision 3 runtime. Add
it through an explicit installer with upstream URL, version, licence and SHA-256
checksum if a future workflow requires it.
