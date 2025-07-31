import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a low-res preview video")
    parser.add_argument("input", help="High resolution input video")
    parser.add_argument("output", help="Output preview video path")
    parser.add_argument("--width", type=int, default=320, help="Output width")
    parser.add_argument("--crf", type=int, default=28, help="ffmpeg CRF value")
    args = parser.parse_args()

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(Path(args.input)),
        "-vf",
        f"scale={args.width}:-2",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        str(args.crf),
        "-c:a",
        "copy",
        str(Path(args.output)),
    ]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
