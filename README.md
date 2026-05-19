# qs-automation

PDF symbol counter for construction quantity takeoffs.

Import PDF drawings, extract a legend, and count symbols across pages automatically. Draw zones to get per-zone counts. Export results to a spreadsheet.

![screenshot placeholder]()

## Features

- Import multi-page PDF drawings
- Auto-extract legend from drawings (OCR-based)
- Manual and automatic marker placement
- Template matching with color pre-filter for symbol counting
- Draw polygon or rectangle zones, get per-zone counts
- Undo/redo for manual edits
- Export counts to xlsx (rows per zone, columns per legend entry)
- Project files saved as `.qsproj` (SQLite)

## Install

Requires Python 3.13.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Stack

- PyQt6 - UI
- PyMuPDF - PDF rendering
- OpenCV - template matching
- EasyOCR + img2table - legend extraction
- SQLAlchemy - project persistence

## Planned

- Text filter: use OCR-detected text positions as an additional spatial pre-filter on match candidates
- Parts-based matching for more accurate detection of multi-component symbols
- LRU page cache to bound memory usage across large projects
- Web port with collaborative markup
