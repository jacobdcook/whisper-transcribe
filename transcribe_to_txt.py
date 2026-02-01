#!/usr/bin/env python3
"""Transcribe audio files to text using faster-whisper (large-v3, CUDA)."""

import argparse
import glob
import os
import sys
import time


def _ensure_cuda_libs():
    """Preload pip-installed NVIDIA CUDA libs so ctranslate2 can find them."""
    import ctypes
    import ctypes.util

    if ctypes.util.find_library("cublas"):
        return  # system CUDA is available, nothing to do

    # Find libs installed by nvidia-cublas-cu12 / nvidia-cudnn-cu12 pip packages
    site_pkgs = os.path.join(sys.prefix, "lib", "python*", "site-packages", "nvidia", "*", "lib")
    nvidia_dirs = glob.glob(site_pkgs)
    if not nvidia_dirs:
        return

    # Preload .so files so dlopen() calls from ctranslate2 will find them
    for d in nvidia_dirs:
        for so in sorted(glob.glob(os.path.join(d, "*.so*"))):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def pick_audio_file() -> str:
    """Open a file-picker dialog and return the selected path."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select an audio file",
        filetypes=[
            ("Audio files", "*.mp3 *.wav *.m4a *.flac *.ogg *.opus *.webm"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path


def format_time(seconds: float) -> str:
    """Format seconds into mm:ss or hh:mm:ss."""
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60}:{seconds % 60:02d}"
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def progress_bar(pct: float, width: int = 30) -> str:
    """Return a text progress bar like [████████░░░░░░░░]."""
    filled = int(width * pct)
    return "█" * filled + "░" * (width - filled)


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio to text using faster-whisper.")
    parser.add_argument("file", nargs="?", help="Path to audio file (opens file picker if omitted)")
    parser.add_argument("--model", default="large-v3", help="Whisper model size (default: large-v3)")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument("--output", "-o", help="Output file path (default: same name as input with .txt)")
    args = parser.parse_args()

    # --- Get audio path ---
    audio_path = args.file
    if not audio_path:
        audio_path = pick_audio_file()
    if not audio_path:
        print("No file selected. Exiting.")
        sys.exit(1)
    if not os.path.isfile(audio_path):
        print(f"File not found: {audio_path}")
        sys.exit(1)

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"Audio file : {os.path.basename(audio_path)} ({file_size_mb:.1f} MB)")

    # --- Load model ---
    _ensure_cuda_libs()
    print(f"Loading model '{args.model}' on CUDA ...", end=" ", flush=True)
    load_start = time.time()
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    print(f"done ({time.time() - load_start:.1f}s)")

    # --- Transcribe with progress ---
    print("Starting transcription ...")
    segments_gen, info = model.transcribe(
        audio_path,
        language=args.language,
        task="transcribe",
        vad_filter=True,
        beam_size=5,
    )

    duration = info.duration
    print(f"Audio length: {format_time(duration)}\n")

    lines: list[str] = []
    start_time = time.time()

    for seg in segments_gen:
        text = seg.text.strip()
        if text:
            lines.append(text)

        pct = min(seg.end / duration, 1.0) if duration > 0 else 0
        elapsed = time.time() - start_time
        eta = (elapsed / pct - elapsed) if pct > 0 else 0

        status = (
            f"\r  {progress_bar(pct)} {pct:5.1%}"
            f"  |  {format_time(seg.end)}/{format_time(duration)}"
            f"  |  elapsed {format_time(elapsed)}"
            f"  |  ETA {format_time(eta)}"
        )
        print(status, end="", flush=True)

    elapsed_total = time.time() - start_time
    print(f"\r  {progress_bar(1.0)} 100.0%  |  done in {format_time(elapsed_total)}                    ")

    # --- Save output ---
    out_path = args.output or (os.path.splitext(audio_path)[0] + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nTranscript saved to: {out_path}")
    print(f"Total segments: {len(lines)}")


if __name__ == "__main__":
    main()
