# Sememes

Sememes is a personal knowledge tool that records voice notes, transcribes with Whisper, auto-translates to English, and turns all your memos into a cross-file **Topic Map** — an interactive graph of recurring themes linked back to the exact moments they appear.

## Features

### Note Management
- **Evernote-inspired layout** — Sidebar navigation, note list, and full editor with breadcrumb bar showing notebook > note path
- **Notebooks** — Organize notes into notebooks with creation dates and note counts; dedicated notebooks list view
- **Home dashboard** — Notebook tiles grid with stats (total notes, notebook count)
- **Create, edit, rename, duplicate, and delete** notes with auto-save (2-second debounce)
- **Trash / restore** — Deleted notes go to trash; restore or permanently delete; "Empty Trash" clears all
- **Sorting** — Sort notes by created date, updated date, or title (ascending/descending)
- **Dense/compact mode** — Toggle compact note list for more notes on screen
- **Rich text editor** — Bold, Italic, Underline, Strikethrough, Headings (H1-H3), Bullet/Numbered lists, Checklists, Code blocks, Blockquotes, Tables
- **Links** — External hyperlinks (Ctrl+K) and internal note links (Ctrl+Shift+K) with autocomplete
- **Image embedding** — Upload and embed images inline
- **File attachments** — Attach any file type to notes

### Search
- **Search overlay** (Ctrl+K) — Evernote-style two-column popup with results on left, preview cards on right
- **Scope filter** — Search everywhere or within a specific notebook
- **Highlighted matches** — Search terms highlighted in titles and content snippets
- **Smart previews** — Context snippets centered around matches
- **In-note search** (Ctrl+F) — Find and replace within the open note with match highlighting and navigation

### Files
- **Files panel** — Full-screen file browser accessible from sidebar
- **Tabs** — Filter by All Files, Media (images/audio/video), or Docs
- **File preview** — Right panel shows image/audio/video/PDF/text/CSV preview
- **CSV as table** — CSV files rendered as formatted, scrollable tables
- **Upload** — Upload standalone files or attach to notes
- **Download, delete, open parent note** from preview panel

### Audio & Transcription
- **Audio upload** — Supports m4a, mp3, wav, ogg, flac, webm, mp4, wma
- **Whisper transcription** — Chunked transcription with real-time SSE progress streaming
- **Auto-translation** — GPT-4o-mini translates transcriptions to English; bilingual tab view

### Import & Export
- **Evernote import** — Import `.enex` files; notebook name derived from filename with manual override prompt; preserves titles, content, dates, tags, and metadata (source URL, author, location)
- **Export** — Single note or bulk export as Markdown, HTML, JSON, or ENEX
- **Text file upload** — Supports txt, md, csv, json, log, rtf
- **Drag-and-drop** — Drop audio, text, or .enex files directly into the app

### Reliability & Safety
- **Version history** — Auto-snapshots before every save; browse and restore previous versions
- **Conflict protection** — Optimistic locking detects concurrent edits; reload or overwrite options
- **Note recovery** — Current state saved before restoring a version; nothing is ever lost

### Navigation
- **Back/forward buttons** — Browser-style navigation history with arrow buttons in sidebar
- **Mouse back/forward** — Mouse side buttons supported
- **Keyboard** — Alt+Left / Alt+Right for back/forward
- **Command palette** (Ctrl+P) — Searchable list of all commands and notebooks

### Topic Map
- **Cross-file analysis** — GPT-4o-mini extracts recurring topics across all notes
- **Full-screen view** — Dedicated full-screen mode from sidebar
- **Interactive graph** — vis.js network with force-directed layout, auto-centered on open
- **Hover highlighting** — Topic node highlights matching keywords in the open note
- **Click navigation** — Click a node to see and jump to files mentioning that topic
- **Smart caching** — MD5 hash invalidation; "stale" badge when library changes

### UX
- **Material Design 3** — M3 color system, typography, shape tokens, and elevation
- **Resizable panels** — Drag handles on panel borders; sizes persisted
- **Collapsible sidebars** — Toggle notebooks sidebar (Ctrl+\\) and notes panel (Ctrl+Shift+\\)
- **Six themes** — Light (M3), Evernote (high contrast green), Dark, Midnight, Warm, Nord

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask + SQLite |
| Transcription | OpenAI Whisper (medium model) |
| AI | GPT-4o-mini (translation + topic analysis) |
| Frontend | Vanilla JS, CSS custom properties, Material Design 3 |
| Graph | vis.js |

## Setup

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Install [ffmpeg](https://ffmpeg.org/) (required for audio conversion)
4. Copy `.env.example` to `.env` and add your `OPENAI_API_KEY`
5. Run: `python app.py`
6. Open http://127.0.0.1:5000

> **Note:** First launch downloads the Whisper medium model (~1.5 GB) and loads it into RAM (~4.6 GB). The server takes ~35 seconds to start.
