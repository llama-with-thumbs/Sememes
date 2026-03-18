# Sememes

Sememes is a personal knowledge tool that records voice notes, transcribes with Whisper, auto-translates to English, and turns all your memos into a cross-file **Topic Map** — an interactive graph of recurring themes linked back to the exact moments they appear.

## Features

### Note Management
- **Three-panel layout** — Notebooks sidebar, note list with search, and full editor (inspired by Evernote)
- **Notebooks** — Organize notes into notebooks; every note belongs to a notebook ("My Notebook" by default)
- **Create, edit, rename, duplicate, and delete** notes with auto-save (2-second debounce)
- **Starred / pinned notes** — Star important notes; they appear first in the list and have a dedicated "Starred" view
- **Trash / restore** — Deleted notes go to trash first; restore or permanently delete from the trash view; "Empty Trash" clears all
- **Recent Notes** — Quick-access view showing the 20 most recently updated notes
- **Sorting** — Sort notes by created date, updated date, or title (ascending/descending)
- **Search** — Real-time search across note titles and content
- **Rich text editor** — Formatting toolbar with Bold, Italic, Underline, Strikethrough, Headings (H1–H3), Bullet/Numbered lists, and a whitespace cleanup tool

### Audio & Transcription
- **Audio upload** — Supports m4a, mp3, wav, ogg, flac, webm, mp4, wma
- **Whisper transcription** — Chunked transcription with real-time SSE progress streaming
- **Auto-translation** — GPT-4o-mini translates transcriptions to English; bilingual tab view (English / Original)

### Import & Upload
- **Evernote import** — Import `.enex` files with full support for titles, content (ENML → HTML), dates, and tags
- **Text file upload** — Supports txt, md, csv, json, log, rtf
- **Drag-and-drop** — Drop audio, text, or .enex files directly into the app

### Topic Map
- **Cross-file analysis** — GPT-4o-mini extracts 5–20 recurring topics across all notes
- **Interactive graph** — vis.js network visualization with force-directed layout
- **Hover highlighting** — Hovering a topic node highlights matching keywords in the open note
- **Click navigation** — Click a node to see which files mention that topic, then jump to any of them
- **Smart caching** — Topic map cached with MD5 hash invalidation; "stale" badge when library changes
- **Collapsible panel** — Hidden by default, toggled via toolbar button

### Themes
Six built-in themes: Light, Evernote (green accent), Dark, Midnight, Warm, and Nord

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Browser (SPA)                              │
│                                                                     │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │  Notebooks  │  │  Note List   │  │   Editor     │  │ Topic Map│ │
│  │   Panel     │  │   Panel      │  │   Panel      │  │  Panel   │ │
│  │             │  │              │  │              │  │(toggle)  │ │
│  │ - Notebook  │  │ - Search     │  │ - Toolbar    │  │          │ │
│  │   list      │  │ - Note cards │  │ - Title      │  │ - vis.js │ │
│  │ - Theme     │  │ - Upload     │  │ - Metadata   │  │   graph  │ │
│  │   selector  │  │ - Import     │  │ - Rich text  │  │ - Node   │ │
│  │             │  │              │  │ - Audio bar  │  │   hover  │ │
│  └────────────┘  └──────────────┘  └──────────────┘  └──────────┘ │
│         │                │                │                │        │
│         └────────────────┴────────────────┴────────────────┘        │
│                          Fetch API + SSE                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP / SSE
┌──────────────────────────────┴──────────────────────────────────────┐
│                        Flask Backend (app.py)                       │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      REST API Routes                         │   │
│  │  /library    /notebooks    /upload    /import-enex           │   │
│  │  /library/<id>/text    /transcribe    /build-topic-map       │   │
│  └──────┬───────────┬──────────────┬───────────────┬────────────┘   │
│         │           │              │               │                │
│  ┌──────┴──┐ ┌──────┴──────┐ ┌────┴────┐ ┌───────┴──────────┐     │
│  │ SQLite  │ │   Whisper   │ │  ffmpeg  │ │  OpenAI GPT-4o   │     │
│  │   DB    │ │  (medium)   │ │ ffprobe  │ │     -mini        │     │
│  │         │ │             │ │          │ │                   │     │
│  │ files   │ │ Audio →     │ │ Audio    │ │ - Translation     │     │
│  │ note-   │ │ Text        │ │ convert  │ │ - Topic map       │     │
│  │ books   │ │ ~4.6GB RAM  │ │ & split  │ │   analysis        │     │
│  │ cache   │ │             │ │          │ │                   │     │
│  └─────────┘ └─────────────┘ └─────────┘ └───────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘

Data Flow:
  Audio Upload → ffmpeg (MP3) → Whisper (transcribe) → GPT (translate) → SQLite
  Evernote .enex → XML parse → ENML→HTML → SQLite
  Topic Map → All transcriptions → GPT (analyze) → vis.js graph
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask + SQLite |
| Transcription | OpenAI Whisper (medium model) |
| AI | GPT-4o-mini (translation + topic analysis) |
| Frontend | Vanilla JS, CSS custom properties |
| Graph | vis.js |

## Setup

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Install [ffmpeg](https://ffmpeg.org/) (required for audio conversion)
4. Copy `.env.example` to `.env` and add your `OPENAI_API_KEY`
5. Run: `python app.py`
6. Open http://127.0.0.1:5000

> **Note:** First launch downloads the Whisper medium model (~1.5 GB) and loads it into RAM (~4.6 GB). The server takes ~35 seconds to start.
