# Inkshade PDF Narration — Standalone POC Implementation Plan

## Project Overview

Build a standalone Python script that takes a PDF file as input and produces a narrated MP3 audio file as output. The narration should sound human-like by leveraging document structure analysis (ML-based layout detection + typographic feature refinement) to drive prosody decisions — pauses, speed changes, and element skipping. The script reuses Inkshade's existing PDF backend for text extraction and page rendering.

After this POC is validated and refined, it will be integrated into the Inkshade PDF Reader GUI with real-time playback, synchronized highlighting, and user controls.

---

## Backend Components to Migrate from Inkshade

These are the specific modules and classes from the existing codebase that the POC depends on. They will be imported directly (not copied/rewritten) so the POC stays in sync with the main app.

### Required As-Is (Direct Import)

| Module | Class/Function | Purpose in POC |
|---|---|---|
| `core/document/pdf_reader.py` | `PDFDocumentReader` | Load the PDF, access `fitz.Document`, get page count, get page objects |
| `core/page/text_layer.py` | `PageTextLayer` | Extract character-level text structure per page — `blocks`, `characters`, `SpanInfo`, `LineInfo` |
| `core/page/models.py` | `BlockInfo`, `LineInfo`, `SpanInfo`, `CharacterInfo` | Data models for text structure |
| `core/page/page_model.py` | `PageModel` | Unified page access — wraps text layer, provides `render_pixmap` for generating page images for the YOLO model |

### Required but Bypassed Partially

| Module | What's Used | What's Skipped |
|---|---|---|
| `core/document/pdf_reader.py` | `load_pdf()`, `get_page()`, `get_page_count()` | All Qt/GUI methods (`render_page` with QPixmap, QMessageBox error dialogs). The POC will call `fitz` directly for image rendering to avoid Qt dependency in the standalone script. |
| `core/page/page_model.py` | `text_layer` property, page dimensions (`width`, `height`) | `render_pixmap()` (Qt-dependent). The POC renders page images via `fitz.Page.get_pixmap()` directly and converts to PIL Image for YOLO. |

### NOT Required for POC

| Module | Reason |
|---|---|
| `core/annotations/*` | No annotation handling needed for narration |
| `core/selection/*` | No text selection needed |
| `core/search/*` | No search functionality needed |
| `core/export/*` | No PDF export needed |
| `controllers/*` | All GUI controllers, not relevant |
| `ui/*` | All GUI components |
| `styles/*` | Theme management |
| `utils/warning_manager.py` | GUI warning dialogs |
| `utils/resource_loader.py` | Icon/resource loading for GUI |

### Key Consideration: Qt Dependency

`PDFDocumentReader` and `PageModel` import `PyQt5` for `QImage`, `QPixmap`, `QMessageBox`. For the standalone POC, there are two options:

- **Option A (Recommended):** Keep the imports but don't call Qt-dependent methods. Only use `fitz` directly for page rendering to PIL Images. This avoids needing a running Qt application instance.
- **Option B:** Create thin wrapper functions that replicate just the text extraction logic without Qt. This is cleaner but means maintaining a parallel code path.

Option A is better for a POC — less code, and when you integrate back into the GUI app, everything is already compatible.

---

## POC Project Structure

```
inkshade-narrate/
├── narrate.py                          # Entry point CLI script
├── core/                               # Symlink or import path to Inkshade core/
│   └── (existing Inkshade core modules)
│
├── narration/
│   ├── __init__.py
│   ├── pipeline.py                     # Orchestrator: PDF → analysis → script → audio → MP3
│   │
│   ├── layout/
│   │   ├── __init__.py
│   │   ├── detector.py                 # YOLOv8 DocLayNet wrapper
│   │   ├── block_matcher.py            # Maps YOLO regions → TextLayer BlockInfo
│   │   ├── feature_refiner.py          # Refines labels using font/position metadata
│   │   └── models.py                   # LayoutLabel, LayoutRegion, ClassifiedBlock
│   │
│   ├── script/
│   │   ├── __init__.py
│   │   ├── reading_script.py           # Builds ordered ReadingInstruction list
│   │   ├── prosody_rules.py            # Role → pause/speed/emphasis mappings
│   │   ├── text_preprocessor.py        # Cleans text, adds natural pause points
│   │   └── models.py                   # ReadingInstruction, ProsodyRule, TextRole
│   │
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── engine.py                   # Piper TTS wrapper
│   │   ├── audio_builder.py            # Concatenates WAV chunks + silence gaps → final audio
│   │   └── model_manager.py            # Downloads/caches Piper voice models
│   │
│   └── utils/
│       ├── __init__.py
│       └── audio_utils.py              # WAV→MP3 conversion, silence generation, normalization
```

---

## Task Breakdown

### Task 0: Environment & Dependency Setup
**Goal:** Prepare the development environment with all required libraries.

- **0.1** Create the `inkshade-narrate/` project directory with the structure above
- **0.2** Set up import paths so `core/` modules from Inkshade are accessible (symlink, sys.path manipulation, or pip editable install of Inkshade)
- **0.3** Install dependencies:
  - `piper-tts` — neural TTS engine
  - `ultralytics` — YOLOv8 inference
  - `Pillow` — image handling for YOLO input
  - `pydub` — audio concatenation and MP3 export
  - `numpy` — audio buffer manipulation
  - `PyMuPDF` (fitz) — already an Inkshade dependency
- **0.4** Verify that `core/page/text_layer.py` and `core/document/pdf_reader.py` can be imported and used without a running QApplication (test with a simple script that loads a PDF and prints block text)
- **0.5** Download and verify a Piper voice model (`en_US-lessac-medium` recommended for starting)
- **0.6** Download or obtain a YOLOv8 model trained on DocLayNet (either find pre-trained weights or fine-tune — see Task 2)

**Deliverable:** A working environment where you can import Inkshade core, run Piper, and run YOLO inference independently.

---

### Task 1: Text Extraction Adapter
**Goal:** Create a clean interface between Inkshade's text extraction and the narration pipeline.

- **1.1** Write a function `extract_page_text_structure(pdf_path, page_index)` that:
  - Opens the PDF using `fitz.open()` (bypass `PDFDocumentReader` to avoid Qt)
  - Creates a `PageTextLayer` from the fitz page
  - Returns the list of `BlockInfo` objects with full character/span/line data
- **1.2** Write a function `extract_all_pages(pdf_path)` that iterates all pages and returns a `Dict[int, List[BlockInfo]]`
- **1.3** Write a function `render_page_to_pil(pdf_path, page_index, scale=1.5)` that:
  - Renders the page to a `fitz.Pixmap`
  - Converts to a `PIL.Image` (not QImage) for YOLO input
  - The scale factor controls resolution — 1.5 is a good balance between detection accuracy and speed
- **1.4** Write a helper `get_page_dimensions(pdf_path, page_index)` returning `(width, height)` in PDF points
- **1.5** Test all four functions on 3-5 diverse PDFs (academic paper, book chapter, slide-style PDF, report with tables/figures)

**Deliverable:** `narration/utils/pdf_adapter.py` (or similar) — a Qt-free interface to Inkshade's text extraction.

---

### Task 2: Layout Detection Module
**Goal:** Detect document layout regions (title, heading, body, footnote, etc.) from page images using YOLO.

- **2.1** Research and obtain a YOLOv8 model trained on DocLayNet. Options:
  - Search HuggingFace for `doclaynet yolov8` — several community-trained models exist
  - If none are satisfactory, fine-tune YOLOv8s on the DocLayNet dataset (IBM provides it freely). This takes a few hours on a GPU but gives you full control.
  - Document which model you chose and its class mapping
- **2.2** Implement `LayoutDetector` class:
  - `__init__(model_path)` — loads the YOLO model
  - `detect(pil_image, confidence=0.35) → List[LayoutRegion]` — runs inference, maps class indices to `LayoutLabel` enum, returns normalized bounding boxes
  - Handle the class index mapping between DocLayNet labels and your `LayoutLabel` enum
- **2.3** Implement model management:
  - `ModelManager.ensure_model_available()` — checks if model exists in app data dir, downloads if not
  - Store models in `~/.local/share/InkshadePDF/models/` (reuse Inkshade's app data convention)
- **2.4** Test the detector on diverse PDFs:
  - Print detected regions with labels and confidence scores
  - Visually verify by drawing bounding boxes on the page image and saving to disk
  - Identify systematic misclassifications to address in the refinement step

**Deliverable:** `narration/layout/detector.py` and `narration/layout/models.py` — working layout detection that produces labeled regions per page.

---

### Task 3: Block Matching & Classification Refinement
**Goal:** Map YOLO detections to text blocks and refine with typographic features.

- **3.1** Implement `match_regions_to_blocks()`:
  - Takes `List[LayoutRegion]` (normalized coords) and `List[BlockInfo]` (PDF coords)
  - Normalizes all bounding boxes to a common coordinate space
  - Computes IoU (Intersection over Union) between each YOLO region and each text block
  - Assigns each block the label of its best-matching region (above an IoU threshold)
  - Blocks with no matching region get `LayoutLabel.UNKNOWN`
  - Handle edge cases: one YOLO region covering multiple blocks (assign same label), one block overlapping multiple regions (pick highest IoU)
- **3.2** Implement `FeatureRefiner`:
  - Compute page-level statistics: median font size, font size distribution, page height
  - For `UNKNOWN` blocks, classify using font size ratio, bold flags, word count, position
  - For low-confidence detections, cross-check against text features:
    - Is the "TEXT" block actually a list item? (check for bullet/number patterns)
    - Is the "SECTION_HEADER" actually a title? (check relative font size)
    - Is the block a page number? (single number near bottom of page)
  - Detect running headers/footers: if the same text appears at the same position on 3+ consecutive pages, mark as `PAGE_HEADER`/`PAGE_FOOTER`
- **3.3** Create the combined function `classify_page(page_image, text_layer, page_width, page_height) → List[ClassifiedBlock]` that chains detection → matching → refinement
- **3.4** Build a visual debugging tool: renders each page with color-coded overlays per label (red = title, blue = heading, green = body, gray = skipped) and saves as images. This is essential for iterating on accuracy.
- **3.5** Test on 5-10 diverse PDFs. For each, manually verify the classifications and note:
  - False positives (body text labeled as heading)
  - False negatives (headings labeled as body)
  - Missed skip targets (page numbers/headers read aloud)
  - Tune confidence thresholds and refinement rules based on findings

**Deliverable:** `narration/layout/block_matcher.py` and `narration/layout/feature_refiner.py` — robust classification pipeline that labels every text block on a page.

---

### Task 4: Reading Script Builder
**Goal:** Convert classified blocks into an ordered sequence of reading instructions with prosody annotations.

- **4.1** Define the data models:
  - `TextRole` enum (maps from `LayoutLabel` — some labels map to the same role, e.g., `PAGE_HEADER` and `PAGE_FOOTER` both map to a "skip" role)
  - `ProsodyRule` dataclass: `pause_before`, `pause_after`, `speed_factor`, `skip`, `emphasis`
  - `ReadingInstruction` dataclass: `text`, `role`, `prosody`, `page_index`, `block_index`, `characters` (for future highlighting)
- **4.2** Define the prosody rules table:
  - Map each `TextRole` to a `ProsodyRule` with tuned defaults
  - Make this a configuration (dict or JSON file) so it's easy to tweak without code changes
  - Include sensible defaults as discussed (titles get 1.5s pause before, 0.85 speed; body gets 0.3s paragraph pause; page numbers are skipped)
- **4.3** Implement `TextPreprocessor`:
  - Clean text for TTS consumption:
    - Remove hyphenation at line breaks (e.g., "com-\nputer" → "computer")
    - Normalize whitespace
    - Expand common abbreviations that TTS might mispronounce (e.g., "Fig." → "Figure", "eq." → "equation") — make this a configurable dictionary
    - Handle special characters: remove or replace dashes, ellipses, etc. with TTS-friendly equivalents
    - Detect and reformat URLs: replace with "link to [domain]" or skip entirely
    - Detect reference markers like "[1]", "[2,3]" and optionally strip them
  - Split body text into sentences for chunk-based processing:
    - Use a robust sentence splitter (regex-based is fine for POC; `nltk.sent_tokenize` if you want better accuracy)
    - Keep sentences mapped to their source block and character positions
  - Add intra-paragraph pause hints:
    - After colons: slightly longer pause
    - After semicolons: moderate pause
    - Before direct quotes: brief pause
- **4.4** Implement `ReadingScriptBuilder`:
  - `build_page_script(classified_blocks, page_index) → List[ReadingInstruction]`
    - Filters out blocks marked as skip (page numbers, headers, footers, figures, tables, formulas)
    - Orders remaining blocks by vertical position (top to bottom), then horizontal (left to right for multi-column layouts)
    - For each block: looks up prosody rule by role, preprocesses text, creates ReadingInstruction(s)
    - Body blocks get split into per-sentence instructions (for chunk-based TTS)
    - Title/heading blocks stay as single instructions (read as one unit)
  - `build_document_script(all_classified_blocks) → List[ReadingInstruction]`
    - Iterates all pages, builds per-page scripts, concatenates
    - Inserts a page transition pause (e.g., 1.0s) between pages
    - Optionally inserts "Page [N]" announcement (configurable, off by default)
- **4.5** Implement a script preview/debug function:
  - Prints the reading script in a human-readable format:
    ```
    [PAGE 1]
    [TITLE, pause_before=1.5s, speed=0.85x] "Introduction to Machine Learning"
    [PAUSE 1.2s]
    [BODY, speed=1.0x] "Machine learning is a subfield of artificial intelligence..."
    [PAUSE 0.3s]
    [BODY, speed=1.0x] "This chapter introduces the fundamental concepts."
    [PAGE_TRANSITION 1.0s]
    [PAGE 2]
    [HEADING, pause_before=1.2s, speed=0.88x] "Supervised Learning"
    ...
    ```
  - This is critical for tuning prosody without waiting for audio generation
- **4.6** Test the script builder on diverse PDFs, review the preview output, and iterate on:
  - Reading order correctness (especially for multi-column layouts)
  - Appropriate skip decisions
  - Pause and speed values that feel natural (compare to how you'd read the document aloud)

**Deliverable:** `narration/script/` package — converts classified blocks into a complete reading script with prosody annotations.

---

### Task 5: TTS Engine Wrapper
**Goal:** Wrap Piper TTS to synthesize text segments with variable speed.

- **5.1** Implement `TTSEngine`:
  - `__init__(voice_model_path)` — loads the Piper voice
  - `synthesize(text, speed_factor=1.0) → bytes` — synthesizes text to raw PCM/WAV bytes
    - Piper's `length_scale` parameter controls speed: `length_scale = 1.0 / speed_factor`
    - Returns raw audio bytes (WAV format or raw PCM with known sample rate)
  - `get_sample_rate() → int` — returns the model's native sample rate (typically 22050 Hz)
  - `get_audio_duration(audio_bytes) → float` — calculates duration in seconds from byte length and sample rate
- **5.2** Implement `ModelManager` for Piper voices:
  - `ensure_voice_available(voice_name)` — downloads voice model + config if not cached
  - `list_available_voices()` — returns locally cached voices
  - `get_voice_path(voice_name) → str` — returns path to the `.onnx` model file
  - Store voices in `~/.local/share/InkshadePDF/voices/`
- **5.3** Test synthesis:
  - Synthesize a title at 0.85x speed, a heading at 0.88x, body at 1.0x, a footnote at 1.05x
  - Listen to each and verify the speed differences are perceptible but not jarring
  - Measure synthesis time per sentence to confirm real-time feasibility

**Deliverable:** `narration/tts/engine.py` and `narration/tts/model_manager.py` — reliable TTS synthesis with speed control.

---

### Task 6: Audio Builder
**Goal:** Assemble synthesized chunks and silence gaps into a single continuous audio file, then export as MP3.

- **6.1** Implement `generate_silence(duration_seconds, sample_rate) → AudioSegment`:
  - Creates a silent audio segment of the specified duration
  - Use `pydub.AudioSegment.silent(duration=ms)`
- **6.2** Implement `AudioBuilder`:
  - `__init__(sample_rate)` — initializes with the TTS sample rate
  - `add_silence(duration_seconds)` — appends silence to the buffer
  - `add_speech(wav_bytes)` — appends synthesized speech audio
  - `get_duration() → float` — returns current total duration in seconds
  - Internal buffer: use `pydub.AudioSegment` for concatenation (handles format differences gracefully)
- **6.3** Implement audio post-processing:
  - `normalize(target_dBFS=-20)` — normalizes volume to a consistent level across the entire file
  - `apply_crossfade(ms=50)` — applies tiny crossfades between segments to eliminate clicks/pops at boundaries
- **6.4** Implement export:
  - `export_mp3(output_path, bitrate="192k")` — exports the assembled audio as MP3
  - `export_wav(output_path)` — exports as WAV (useful for debugging, no quality loss)
  - Print total duration and file size on export
- **6.5** Test the builder:
  - Manually create a sequence: 1s silence → synthesized sentence → 0.5s silence → synthesized sentence → export
  - Verify no clicks, consistent volume, correct total duration

**Deliverable:** `narration/tts/audio_builder.py` and `narration/utils/audio_utils.py` — assembles and exports the final audio file.

---

### Task 7: Pipeline Orchestrator
**Goal:** Wire everything together into a single function: `pdf_to_narration(pdf_path, output_mp3_path)`.

- **7.1** Implement `NarrationPipeline`:
  - `__init__(voice_name, prosody_config)` — initializes all components (detector, TTS engine, script builder)
  - `narrate(pdf_path, output_path, page_range=None)` — the main entry point
- **7.2** The `narrate()` method follows this sequence:
  ```
  1. Open PDF, get page count
  2. For each page in range:
     a. Render page to PIL Image
     b. Extract text structure (PageTextLayer)
     c. Run layout detection (YOLO)
     d. Match regions to blocks
     e. Refine classifications
     f. Build reading script for page
  3. Concatenate all page scripts into document script
  4. For each ReadingInstruction in script:
     a. If skip → continue
     b. Add silence (pause_before) to AudioBuilder
     c. Synthesize text with speed_factor → WAV bytes
     d. Add speech to AudioBuilder
     e. Add silence (pause_after) to AudioBuilder
     f. Print progress: "[Page X] [ROLE] text preview..."
  5. Post-process audio (normalize, crossfade)
  6. Export to MP3
  7. Print summary: total pages, total instructions, skipped elements, duration, file size
  ```
- **7.3** Add progress reporting:
  - Print per-page progress: `Processing page 3/48...`
  - Print per-phase progress within a page: `Detecting layout... Classifying... Building script... Synthesizing...`
  - Print running time estimates
- **7.4** Add error handling:
  - Pages that fail layout detection: fall back to feature-only classification
  - Pages with no extractable text (scanned images): skip with warning
  - TTS failures on specific sentences: skip sentence, log warning, continue
  - Catch and report but don't crash on any single-page failure
- **7.5** Add configuration options:
  - `page_range`: tuple of (start, end) for partial processing
  - `voice_name`: Piper voice to use
  - `speed_multiplier`: global speed adjustment (stacks with per-role speed factors)
  - `pause_multiplier`: global pause duration scaling
  - `skip_footnotes`: bool (default True)
  - `skip_captions`: bool (default False)
  - `announce_pages`: bool (default False — whether to say "Page N" between pages)

**Deliverable:** `narration/pipeline.py` — the complete orchestrator.

---

### Task 8: CLI Entry Point
**Goal:** Create a command-line script for running the narration pipeline.

- **8.1** Implement `narrate.py` CLI:
  ```
  python narrate.py input.pdf output.mp3 [options]
  
  Options:
    --voice NAME          Piper voice name (default: en_US-lessac-medium)
    --pages START-END     Page range, e.g., 1-10 (default: all)
    --speed FLOAT         Global speed multiplier (default: 1.0)
    --pause-scale FLOAT   Pause duration multiplier (default: 1.0)
    --skip-footnotes      Skip footnotes (default: on)
    --no-skip-footnotes   Read footnotes
    --skip-captions       Skip figure/table captions
    --announce-pages      Say "Page N" between pages
    --debug-layout DIR    Save layout debug images to directory
    --debug-script        Print the reading script without generating audio
    --output-wav          Export as WAV instead of MP3
  ```
- **8.2** Implement argument parsing with `argparse`
- **8.3** Add a `--debug-layout` mode that runs only the layout detection + classification and saves color-coded overlay images (no TTS) — essential for tuning
- **8.4** Add a `--debug-script` mode that runs layout + script building and prints the reading script preview (no TTS) — essential for tuning prosody
- **8.5** Both debug modes should be usable independently from the full pipeline so you can iterate quickly without waiting for audio synthesis

**Deliverable:** `narrate.py` — a complete CLI tool for PDF narration.

---

### Task 9: Testing & Tuning
**Goal:** Validate quality across document types and tune parameters.

- **9.1** Assemble a test corpus of 10+ PDFs covering:
  - Academic paper (two-column, abstract, references, equations)
  - Book chapter (title page, headings, body, footnotes)
  - Technical report (cover page, TOC, figures, tables, appendices)
  - Slide-style PDF (large fonts, sparse text, images)
  - Legal/government document (dense text, numbered sections)
  - Magazine/marketing PDF (creative layouts, sidebars, pull quotes)
  - Scanned PDF with OCR text layer
- **9.2** Run `--debug-layout` on all test PDFs, review the classification overlays, and iterate on:
  - YOLO confidence threshold
  - IoU threshold for block matching
  - Feature refinement heuristics
- **9.3** Run `--debug-script` on all test PDFs, review the reading scripts, and iterate on:
  - Reading order (especially multi-column)
  - Skip decisions (are the right things being skipped?)
  - Prosody values (do the pauses and speeds feel right when you imagine hearing them?)
- **9.4** Generate full audio for 3-5 key test PDFs, listen end to end, and note:
  - Places where pacing feels wrong
  - Transitions that are too abrupt or too slow
  - Content that should have been skipped
  - Content that was skipped but shouldn't have been
  - Pronunciation issues (domain-specific terms, acronyms)
- **9.5** Tune prosody rules and text preprocessing based on findings
- **9.6** Document known limitations and edge cases for future work

**Deliverable:** Tuned configuration, test results documentation, known issues list.

---

## Execution Order & Dependencies

```
Task 0: Environment Setup
  │
  ├──→ Task 1: Text Extraction Adapter (no dependencies beyond Task 0)
  │
  ├──→ Task 2: Layout Detection Module (no dependencies beyond Task 0)
  │
  └──→ Task 5: TTS Engine Wrapper (no dependencies beyond Task 0)
       │
       └──→ Task 6: Audio Builder (depends on Task 5 for sample rate / WAV format)

Task 1 + Task 2 ──→ Task 3: Block Matching & Refinement

Task 3 ──→ Task 4: Reading Script Builder

Task 4 + Task 6 ──→ Task 7: Pipeline Orchestrator

Task 7 ──→ Task 8: CLI Entry Point

Task 8 ──→ Task 9: Testing & Tuning
```

**Parallelizable work:** Tasks 1, 2, and 5 can all be developed simultaneously after Task 0 is complete. Task 6 can be developed as soon as Task 5 is done.

---

## Future Integration Notes (Post-POC)

Once the POC produces good audio files, the integration into Inkshade GUI will involve:

- Moving `narration/` into the Inkshade project as a new top-level package
- Replacing the sequential pipeline with a `QThread`-based worker that synthesizes page-by-page
- Adding real-time audio playback via `QMediaPlayer` or `sounddevice` instead of writing to file
- Adding a `ReadAloudHighlighter` that paints the current sentence/block on `InteractivePageLabel` using the character position data from `ReadingInstruction`
- Adding UI controls: play/pause/stop button in toolbar, speed slider, voice selector
- Pre-buffering: analyze and synthesize the next page while the current page is playing
- Auto-scroll: advance the viewport as reading progresses across pages

These are all additive changes — the narration logic itself won't change, only the I/O layer (file export → real-time playback).
