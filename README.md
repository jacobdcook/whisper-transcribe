# whisper-transcribe

Fast local audio-to-text transcription using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with CUDA GPU acceleration.

Transcribes audio files to `.txt` with a real-time progress bar, ETA, and elapsed time. Uses the `large-v3` model by default for best accuracy.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on RTX 4090)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Pass a file directly
python transcribe_to_txt.py audio.mp3

# Or run without arguments to open a file picker
python transcribe_to_txt.py

# Options
python transcribe_to_txt.py audio.mp3 --model medium --language es -o output.txt
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `large-v3` | Whisper model (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `--language` | `en` | Language code |
| `-o`, `--output` | `<input>.txt` | Output file path |

## Example output

```
Audio file : interview.mp3 (45.2 MB)
Loading model 'large-v3' on CUDA ... done (2.1s)
Starting transcription ...
Audio length: 32:15

  ██████████████████████████████ 100.0%  |  done in 2:48

Transcript saved to: interview.txt
Total segments: 412
```
