"""Minimal CLI; experiment commands are implemented phase-by-phase."""

import argparse
import platform
import shutil


def main() -> None:
    """Run the selected safe command (no GPU model starts automatically)."""
    parser = argparse.ArgumentParser(description="gpu_my_ocr benchmark controller")
    parser.add_argument("command", choices=["doctor"], help="Inspect local prerequisites")
    args = parser.parse_args()
    if args.command == "doctor":
        print(f"Platform: {platform.platform()}")
        print(f"nvidia-smi: {shutil.which('nvidia-smi') or 'NOT FOUND'}")
        print("Next: see docs/SETUP_FEDORA.md (CUDA framework checks not yet implemented)")


if __name__ == "__main__":
    main()
