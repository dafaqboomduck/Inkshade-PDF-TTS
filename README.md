# Inkshade PDF Narration

Convert PDF documents into narrated audio files using ML-based layout detection and prosody-aware text-to-speech synthesis.

The pipeline analyses document structure — titles, headings, body text, footnotes, page furniture — and uses that understanding to drive natural-sounding narration with contextual pauses, speed variation, and intelligent skipping of non-narrative elements.

## How It Works

```
PDF ──► Page Rendering ──► YOLO Layout Detection ──► Block Classification
                                                            │
        MP3/WAV ◄── Audio Assembly ◄── Piper TTS ◄── Reading Script
```

1. **Layout Detection** — Each page is rendered to an image and fed through a YOLOv8 model trained on [DocLayNet](https://github.com/DS4SD/DocLayNet) to detect structural regions (titles, headings, body, footnotes, tables, figures, page headers/footers).

2. **Block Classification** — Detected regions are matched to text blocks extracted by PyMuPDF using IoU and overlap-ratio heuristics. A typographic feature refiner then cross-checks and corrects labels using font size ratios, bold flags, position, and word count. Running headers and footers are detected across pages.

3. **Reading Script** — Classified blocks are converted into an ordered sequence of reading instructions with prosody annotations. Body text is split into sentences. Each instruction carries a semantic role that maps to tuneable pause durations and speed factors. Page numbers, headers, footers, tables, figures, and formulas are skipped.

4. **TTS Synthesis** — Each instruction is synthesised to audio via [Piper](https://github.com/rhasspy/piper), a fast local neural TTS engine. Per-role speed factors produce slower, more deliberate delivery for titles and headings, normal pace for body text, and slightly faster reading for footnotes.

5. **Audio Assembly** — Speech chunks and silence gaps are concatenated, volume-normalised, and exported as MP3 or WAV.

## Quick Start

```bash
# Basic narration — produces paper.mp3
python narrate.py paper.pdf paper.mp3

# Narrate pages 1–12 at 1.1x speed
python narrate.py book.pdf chapter1.mp3 --pages 1-12 --speed 1.1

# Preview the reading script without generating audio
python narrate.py report.pdf --debug-script --pages 1-5

# Save layout debug images for visual review
python narrate.py paper.pdf --debug-layout debug/paper/ -v 2
```

## Installation

### Requirements

- Python 3.10+
- A YOLOv8 model trained on DocLayNet (see [Model Setup](#model-setup))

### Dependencies

```bash
pip install pymupdf pillow ultralytics piper-tts pydub numpy tqdm
```

**FFmpeg** is required by pydub for MP3 export:

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Arch
sudo pacman -S ffmpeg
```

### Model Setup

The pipeline requires a YOLOv8 model trained on the DocLayNet dataset. Place the weights file at `models/yolov8x_doclaynet.pt` (the default path), or specify a custom path with `--yolo-model`.

Several community-trained models are available on HuggingFace — search for `doclaynet yolov8`. Alternatively, fine-tune your own on the [DocLayNet dataset](https://github.com/DS4SD/DocLayNet) provided by IBM.

### Voice Setup

Piper voice models are downloaded automatically on first use. The default voice is `en_US-lessac-medium`. Available voices:

| Voice | Language | Quality |
|---|---|---|
| `en_US-lessac-medium` | English (US) | Medium |
| `en_US-lessac-high` | English (US) | High |
| `en_US-amy-medium` | English (US) | Medium |
| `en_US-ryan-medium` | English (US) | Medium |
| `en_GB-alan-medium` | English (GB) | Medium |

```bash
# List cached and downloadable voices
python narrate.py --list-voices
```

Voice models are cached in `~/.local/share/InkshadePDF/voices/`.

## CLI Reference

```
python narrate.py input.pdf [output.mp3] [options]
```

### Voice

| Flag | Default | Description |
|---|---|---|
| `--voice NAME` | `en_US-lessac-medium` | Piper voice model |
| `--list-voices` | | List voices and exit |

### Page Range

| Flag | Default | Description |
|---|---|---|
| `--pages N-M` | all | Page range (1-based, inclusive) |

### Prosody

| Flag | Default | Description |
|---|---|---|
| `--speed FLOAT` | `1.0` | Global speed multiplier |
| `--pause-scale FLOAT` | `1.0` | Pause duration multiplier |

### Content

| Flag | Default | Description |
|---|---|---|
| `--skip-footnotes` | on | Skip footnotes |
| `--no-skip-footnotes` | | Read footnotes aloud |
| `--skip-captions` | off | Skip figure/table captions |
| `--keep-references` | off | Keep `[N]` citation markers |
| `--announce-pages` | off | Insert "Page N" between pages |

### Audio

| Flag | Default | Description |
|---|---|---|
| `--output-wav` | | Export as WAV instead of MP3 |
| `--bitrate` | `192k` | MP3 bitrate |

### Layout Model

| Flag | Default | Description |
|---|---|---|
| `--yolo-model PATH` | `models/yolov8x_doclaynet.pt` | YOLO weights file |
| `--yolo-device DEVICE` | auto | Force device (`cpu`, `cuda:0`) |
| `--confidence FLOAT` | `0.35` | Detection confidence threshold |

### Debug & Output

| Flag | Default | Description |
|---|---|---|
| `-v`, `--verbose` | `1` | `0`=quiet, `1`=normal, `2`=debug |
| `--no-progress` | | Disable tqdm progress bars |
| `--debug-layout DIR` | | Save layout overlay images |
| `--debug-script` | | Print reading script, skip audio |

## Project Structure

```
inkshade-narrate/
├── narrate.py                         # CLI entry point
├── core/                              # PDF backend (PyMuPDF-based)
│   ├── document/
│   │   └── pdf_reader.py              # Document loading and page access
│   └── page/
│       ├── models.py                  # BlockInfo, LineInfo, SpanInfo, CharacterInfo
│       ├── page_model.py              # Lazy-loaded page wrapper
│       └── text_layer.py              # Character-level text extraction
│
├── narration/
│   ├── pipeline.py                    # Orchestrator: PDF → audio
│   ├── layout/
│   │   ├── detector.py                # YOLOv8 DocLayNet wrapper
│   │   ├── block_matcher.py           # YOLO regions → text blocks (IoU matching)
│   │   ├── feature_refiner.py         # Typographic cross-check and correction
│   │   ├── classifier.py              # High-level classify_page / classify_document
│   │   └── models.py                  # LayoutLabel, LayoutRegion, ClassifiedBlock
│   ├── script/
│   │   ├── reading_script.py          # Builds ordered ReadingInstruction list
│   │   ├── prosody_rules.py           # TextRole → pause/speed mappings
│   │   ├── text_preprocessor.py       # Cleans text, expands abbreviations, splits sentences
│   │   └── models.py                  # TextRole, ProsodyRule, ReadingInstruction
│   ├── tts/
│   │   ├── engine.py                  # Piper TTS wrapper with speed control
│   │   ├── audio_builder.py           # WAV chunk concatenation and MP3 export
│   │   └── model_manager.py           # Voice model download and caching
│   └── utils/
│       └── pdf_adapter.py             # Qt-free PDF access (stateless + PDFAdapter class)
│
├── tests/
│   ├── test_pdf_adapter.py            # Text extraction validation
│   ├── test_layout_detector.py        # YOLO detection visual test
│   ├── test_classifier.py             # Full classification pipeline visual test
│   ├── test_reading_script.py         # Script preview (no audio)
│   ├── test_tts_engine.py             # TTS synthesis + timing
│   └── test_audio_builder.py          # Audio assembly + export
│
├── verify_core.py                     # Smoke test for core imports
├── models/                            # YOLO weights (gitignored)
└── debug/                             # Debug output (gitignored)
```

## Prosody Defaults

Each text role maps to tuneable prosody parameters. These can be adjusted globally with `--speed` and `--pause-scale`, or modified in `narration/script/prosody_rules.py`.

| Role | Pause Before | Pause After | Speed | Behaviour |
|---|---|---|---|---|
| Title | 1.5s | 1.2s | 0.85× | Slow, deliberate |
| Section Header | 1.2s | 0.8s | 0.88× | Slightly slow |
| Body | — | 0.3s | 1.00× | Normal pace |
| List Item | 0.2s | 0.25s | 0.97× | Slight pause between items |
| Caption | 0.6s | 0.6s | 0.95× | Set apart from body |
| Footnote | 0.8s | 0.5s | 1.05× | Slightly faster |
| Formula | — | — | — | Skipped |
| Page Header/Footer | — | — | — | Skipped |
| Picture / Table | — | — | — | Skipped |
| Page Transition | — | 1.0s | — | Silence between pages |

Inter-sentence pause within body paragraphs: **0.15s**.

## Text Preprocessing

Raw PDF text is cleaned before synthesis to improve TTS output:

- **Hyphenation** — Line-break hyphens are rejoined (`"com-\nputer"` → `"computer"`).
- **Abbreviations** — Common abbreviations are expanded (`"Fig."` → `"Figure"`, `"e.g."` → `"for example"`, `"et al."` → `"and others"`).
- **URLs** — Replaced with `"link to [domain]"`.
- **Citation markers** — `[1]`, `[2,3]` patterns are stripped (configurable).
- **Dashes and ellipses** — Normalised for consistent TTS pronunciation.

The abbreviation dictionary and all preprocessing rules are in `narration/script/text_preprocessor.py`.

## Tests

Each test module is a standalone script that validates one pipeline stage. Run them with `python -m`:

```bash
# Verify core imports work without Qt
python verify_core.py sample.pdf

# Text extraction
python -m tests.test_pdf_adapter sample.pdf

# Layout detection (saves annotated images to debug/layout/)
python -m tests.test_layout_detector sample.pdf --pages 0-4

# Full classification (saves overlay images to debug/classified/)
python -m tests.test_classifier sample.pdf --pages 0-4

# Reading script preview (no audio)
python -m tests.test_reading_script sample.pdf --pages 0-4

# TTS synthesis timing (saves WAV samples to debug/tts/)
python -m tests.test_tts_engine

# Audio assembly (saves test MP3/WAV to debug/audio/)
python -m tests.test_audio_builder
```

## Future Work

This is a standalone proof-of-concept. Planned integration into the Inkshade PDF Reader GUI includes:

- Real-time playback with `QThread`-based synthesis worker
- Synchronised text highlighting on the page view
- Play/pause/stop controls, speed slider, and voice selector
- Page-ahead pre-buffering for seamless playback
- Auto-scroll as narration progresses across pages

## License

Part of the [Inkshade PDF Reader](https://github.com/dafaqboomduck/Inkshade-PDF) project.
