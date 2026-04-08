import hashlib
import json
import mimetypes
import os
import re
import shutil
import sqlite3
import subprocess
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime

import whisper
from dotenv import load_dotenv
from flask import Flask, Response, request, jsonify, render_template, send_file
from openai import OpenAI

from db import get_db, init_db_postgres, is_postgres
import storage

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max upload

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = storage.UPLOAD_FOLDER
LIBRARY_FOLDER = storage.LIBRARY_FOLDER
ATTACHMENTS_FOLDER = storage.ATTACHMENTS_FOLDER

ALLOWED_EXTENSIONS = {"m4a", "mp3", "wav", "ogg", "flac", "webm", "mp4", "wma"}
TEXT_EXTENSIONS = {"txt", "md", "csv", "json", "log", "rtf"}

CHUNK_SECONDS = 120  # 2-minute chunks for progress tracking
DEFAULT_NOTEBOOK = "My Notebook"


# --- Database ---

DB_PATH = os.path.join(BASE_DIR, "library.db")

def init_db():
    if is_postgres():
        conn = get_db()
        init_db_postgres(conn)
        # Ensure default notebooks
        conn.execute(
            "INSERT INTO notebooks (name, created_at) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (DEFAULT_NOTEBOOK, datetime.now().isoformat())
        )
        conn.execute(
            "INSERT INTO notebooks (name, created_at) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            ("Inbox", datetime.now().isoformat())
        )
        conn.execute(
            "UPDATE files SET notebook = %s WHERE notebook IS NULL OR notebook = ''",
            (DEFAULT_NOTEBOOK,)
        )
        conn.commit()
        conn.close()
        return
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id TEXT PRIMARY KEY,
            original_name TEXT,
            mp3_path TEXT,
            duration REAL,
            transcription TEXT,
            transcription_en TEXT,
            graph_json TEXT,
            created_at TEXT
        )
    """)
    # Migrate: add transcription_en if missing
    cols = [r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()]
    if "transcription_en" not in cols:
        conn.execute("ALTER TABLE files ADD COLUMN transcription_en TEXT")
    if "insights_json" not in cols:
        conn.execute("ALTER TABLE files ADD COLUMN insights_json TEXT")
    if "updated_at" not in cols:
        conn.execute("ALTER TABLE files ADD COLUMN updated_at TEXT")
    if "tags" not in cols:
        conn.execute("ALTER TABLE files ADD COLUMN tags TEXT")
    if "notebook" not in cols:
        conn.execute("ALTER TABLE files ADD COLUMN notebook TEXT")
    if "starred" not in cols:
        conn.execute("ALTER TABLE files ADD COLUMN starred INTEGER DEFAULT 0")
    if "trashed_at" not in cols:
        conn.execute("ALTER TABLE files ADD COLUMN trashed_at TEXT")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notebooks (
            name TEXT PRIMARY KEY,
            created_at TEXT
        )
    """)
    # Ensure default notebook and Inbox exist, assign orphan notes
    INBOX_NOTEBOOK = "Inbox"
    conn.execute(
        "INSERT OR IGNORE INTO notebooks (name, created_at) VALUES (?, ?)",
        (DEFAULT_NOTEBOOK, datetime.now().isoformat())
    )
    conn.execute(
        "INSERT OR IGNORE INTO notebooks (name, created_at) VALUES (?, ?)",
        (INBOX_NOTEBOOK, datetime.now().isoformat())
    )
    conn.execute(
        "UPDATE files SET notebook = ? WHERE notebook IS NULL OR notebook = ''",
        (DEFAULT_NOTEBOOK,)
    )
    # Migrate: add stack column to notebooks
    nb_cols = [r[1] for r in conn.execute("PRAGMA table_info(notebooks)").fetchall()]
    if "stack" not in nb_cols:
        conn.execute("ALTER TABLE notebooks ADD COLUMN stack TEXT")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS saved_searches (
            id TEXT PRIMARY KEY,
            name TEXT,
            query TEXT,
            filters TEXT,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attachments (
            id TEXT PRIMARY KEY,
            file_id TEXT,
            filename TEXT,
            mime_type TEXT,
            size INTEGER,
            created_at TEXT,
            FOREIGN KEY (file_id) REFERENCES files(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT,
            file_hash TEXT,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS note_versions (
            id TEXT PRIMARY KEY,
            file_id TEXT,
            title TEXT,
            transcription TEXT,
            transcription_en TEXT,
            created_at TEXT,
            source TEXT DEFAULT 'autosave',
            FOREIGN KEY (file_id) REFERENCES files(id)
        )
    """)
    conn.commit()
    conn.close()


def compute_library_hash(conn):
    rows = conn.execute(
        "SELECT id, length(COALESCE(transcription_en, transcription, '')) as tlen "
        "FROM files WHERE transcription IS NOT NULL ORDER BY id"
    ).fetchall()
    fingerprint = "|".join(f"{r['id']}:{r['tlen']}" for r in rows)
    return hashlib.md5(fingerprint.encode()).hexdigest()


init_db()


# --- Whisper Model ---

print("Loading Whisper 'medium' model (this may take a while on first run)...")
model = whisper.load_model("medium")
print("Model loaded.")


# --- Audio Helpers ---

def get_audio_duration(filepath):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", filepath],
        capture_output=True, encoding="utf-8", errors="replace"
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def split_audio(filepath, chunk_secs=CHUNK_SECONDS):
    """Split audio into chunks, return list of chunk file paths."""
    duration = get_audio_duration(filepath)
    chunks = []
    start = 0
    i = 0
    while start < duration:
        chunk_path = filepath + f".chunk{i}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", filepath, "-ss", str(start), "-t", str(chunk_secs),
             "-ar", "16000", "-ac", "1", chunk_path],
            capture_output=True
        )
        chunks.append(chunk_path)
        start += chunk_secs
        i += 1
    return chunks, duration


def convert_to_mp3(input_path, output_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-b:a", "128k", output_path],
        capture_output=True
    )


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in (ALLOWED_EXTENSIONS | TEXT_EXTENSIONS)


def is_text_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in TEXT_EXTENSIONS


# --- Translation Helper ---

def translate_to_english(text):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Translate the following text to English. If it is already in English, return it as-is. Preserve the original meaning and tone. Return only the translation, nothing else.",
            },
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content


# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Upload audio to library without transcribing."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        all_ext = sorted(ALLOWED_EXTENSIONS | TEXT_EXTENSIONS)
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(all_ext)}"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        file_id = str(uuid.uuid4())

        if is_text_file(file.filename):
            # Text file: read content and store as a text note
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                text_content = f.read()
            conn = get_db()
            conn.execute(
                "INSERT INTO files (id, original_name, mp3_path, duration, transcription, created_at, notebook) VALUES (?, ?, NULL, 0, ?, ?, ?)",
                (file_id, file.filename, text_content, datetime.now().isoformat(), DEFAULT_NOTEBOOK)
            )
            conn.commit()
            conn.close()
            return jsonify({"file_id": file_id, "original_name": file.filename, "duration": 0})

        # Audio file: convert to MP3
        duration = get_audio_duration(filepath)
        mp3_filename = file_id + ".mp3"
        mp3_path = os.path.join(LIBRARY_FOLDER, mp3_filename)
        convert_to_mp3(filepath, mp3_path)

        conn = get_db()
        conn.execute(
            "INSERT INTO files (id, original_name, mp3_path, duration, created_at, notebook) VALUES (?, ?, ?, ?, ?, ?)",
            (file_id, file.filename, mp3_filename, duration, datetime.now().isoformat(), DEFAULT_NOTEBOOK)
        )
        conn.commit()
        conn.close()
        return jsonify({"file_id": file_id, "original_name": file.filename, "duration": duration})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/library/<file_id>/transcribe", methods=["POST"])
def transcribe_library_file(file_id):
    """Transcribe an existing library file via SSE."""
    conn = get_db()
    row = conn.execute("SELECT mp3_path, original_name FROM files WHERE id = ?", (file_id,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "File not found"}), 404

    mp3_full = os.path.join(LIBRARY_FOLDER, row["mp3_path"])

    def generate():
        chunks = []
        try:
            yield sse_event("status", {"message": "Analyzing audio..."})
            chunk_paths, duration = split_audio(mp3_full)
            total_chunks = len(chunk_paths)

            yield sse_event("status", {
                "message": f"Audio is {format_duration(duration)}. Transcribing in {total_chunks} chunks...",
                "duration": duration,
                "total_chunks": total_chunks,
            })

            full_text = ""
            for i, chunk_path in enumerate(chunk_paths):
                chunks.append(chunk_path)
                chunk_start = i * CHUNK_SECONDS
                chunk_end = min((i + 1) * CHUNK_SECONDS, duration)

                yield sse_event("progress", {
                    "chunk": i + 1,
                    "total_chunks": total_chunks,
                    "percent": round((i / total_chunks) * 100),
                    "message": f"Transcribing {format_duration(chunk_start)} - {format_duration(chunk_end)}...",
                    "text_so_far": full_text,
                })

                result = model.transcribe(chunk_path)
                full_text += result["text"] + " "

            full_text = full_text.strip()

            yield sse_event("status", {"message": "Translating to English..."})
            text_en = translate_to_english(full_text)

            conn2 = get_db()
            conn2.execute(
                "UPDATE files SET transcription = ?, transcription_en = ? WHERE id = ?",
                (full_text, text_en, file_id)
            )
            conn2.commit()
            conn2.close()

            yield sse_event("done", {
                "text": full_text,
                "text_en": text_en,
                "file_id": file_id,
                "percent": 100,
            })

        finally:
            for cp in chunks:
                if os.path.exists(cp):
                    os.remove(cp)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/library")
def library_list():
    conn = get_db()
    notebook_filter = request.args.get("notebook")
    search_q = request.args.get("q")
    sort_by = request.args.get("sort", "created_desc")
    view = request.args.get("view")  # "trash", "starred", "recent"
    tag_filter = request.args.get("tag")
    search_field = request.args.get("field")  # "title", "content", or None (all)
    date_from = request.args.get("date_from")  # ISO date string
    date_to = request.args.get("date_to")  # ISO date string

    query = ("SELECT id, original_name, duration, created_at, updated_at, tags, notebook, starred, trashed_at, "
             "COALESCE(transcription_en, transcription, '') as raw_text, "
             "transcription IS NOT NULL as has_transcription FROM files")
    params = []
    conditions = []

    if view == "trash":
        conditions.append("trashed_at IS NOT NULL")
    else:
        conditions.append("trashed_at IS NULL")
        if view == "starred":
            conditions.append("starred = 1")
        if notebook_filter:
            conditions.append("notebook = ?")
            params.append(notebook_filter)

    if tag_filter:
        conditions.append("tags LIKE ?")
        params.append(f'%"{tag_filter}"%')

    if date_from:
        conditions.append("created_at >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("created_at <= ?")
        params.append(date_to + " 23:59:59")

    if search_q:
        s = f"%{search_q}%"
        if search_field == "title":
            conditions.append("original_name LIKE ?")
            params.append(s)
        elif search_field == "content":
            conditions.append("(transcription LIKE ? OR transcription_en LIKE ?)")
            params.extend([s, s])
        else:
            conditions.append("(original_name LIKE ? OR transcription LIKE ? OR transcription_en LIKE ?)")
            params.extend([s, s, s])

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    # Sorting — starred notes first, then by chosen sort
    sort_map = {
        "created_desc": "created_at DESC",
        "created_asc": "created_at ASC",
        "updated_desc": "COALESCE(updated_at, created_at) DESC",
        "updated_asc": "COALESCE(updated_at, created_at) ASC",
        "title_asc": "original_name COLLATE NOCASE ASC",
        "title_desc": "original_name COLLATE NOCASE DESC",
    }
    order = sort_map.get(sort_by, "created_at DESC")
    if view == "trash":
        query += f" ORDER BY trashed_at DESC"
    elif view == "recent":
        query += f" ORDER BY COALESCE(updated_at, created_at) DESC LIMIT 20"
    else:
        query += f" ORDER BY starred DESC, {order}"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    result = []
    for r in rows:
        d = dict(r)
        raw = d.pop("raw_text", "")
        plain = re.sub(r'<[^>]+>', '', raw).strip()
        # Smart preview: show context around match when searching
        if search_q and plain:
            idx = plain.lower().find(search_q.lower())
            if idx >= 0:
                start = max(0, idx - 60)
                end = min(len(plain), idx + len(search_q) + 90)
                snippet = plain[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(plain):
                    snippet = snippet + "..."
                d["preview"] = snippet
            else:
                d["preview"] = plain[:150]
        else:
            d["preview"] = plain[:150]
        result.append(d)
    return jsonify(result)


@app.route("/notebooks")
def list_notebooks():
    conn = get_db()
    nb_rows = conn.execute(
        "SELECT n.name, n.stack, n.created_at, COALESCE(c.cnt, 0) as note_count FROM notebooks n "
        "LEFT JOIN (SELECT notebook, COUNT(*) as cnt FROM files "
        "WHERE notebook IS NOT NULL AND notebook != '' AND trashed_at IS NULL "
        "GROUP BY notebook) c ON c.notebook = n.name "
        "ORDER BY n.name"
    ).fetchall()
    file_nbs = conn.execute(
        "SELECT notebook as name, COUNT(*) as note_count FROM files "
        "WHERE notebook IS NOT NULL AND notebook != '' AND trashed_at IS NULL "
        "GROUP BY notebook ORDER BY notebook"
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) as c FROM files WHERE trashed_at IS NULL").fetchone()["c"]
    starred = conn.execute("SELECT COUNT(*) as c FROM files WHERE starred = 1 AND trashed_at IS NULL").fetchone()["c"]
    trash_count = conn.execute("SELECT COUNT(*) as c FROM files WHERE trashed_at IS NOT NULL").fetchone()["c"]
    conn.close()

    known = {r["name"] for r in nb_rows}
    notebooks = [dict(r) for r in nb_rows]
    for r in file_nbs:
        if r["name"] not in known:
            notebooks.append(dict(r))
    notebooks.sort(key=lambda x: x["name"])
    return jsonify({"total": total, "starred": starred, "trash_count": trash_count, "notebooks": notebooks})


@app.route("/notebooks", methods=["POST"])
def create_notebook():
    data = request.get_json()
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name required"}), 400
    conn = get_db()
    try:
        conn.execute("INSERT INTO notebooks (name, created_at) VALUES (?, ?)",
                     (name, datetime.now().isoformat()))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "Notebook already exists"}), 400
    conn.close()
    return jsonify({"ok": True, "name": name})


@app.route("/notebooks/delete", methods=["POST"])
def delete_notebook():
    data = request.get_json()
    name = data.get("name", "")
    conn = get_db()
    conn.execute("UPDATE files SET notebook = ? WHERE notebook = ?", (DEFAULT_NOTEBOOK, name))
    conn.execute("DELETE FROM notebooks WHERE name = ?", (name,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/library/note", methods=["POST"])
def create_note():
    """Create a text-only note (no audio)."""
    data = request.get_json() or {}
    notebook = data.get("notebook") or DEFAULT_NOTEBOOK
    file_id = str(uuid.uuid4())
    base_name = f"Note {datetime.now().strftime('%Y-%m-%d')}"
    conn = get_db()
    existing = conn.execute(
        "SELECT original_name FROM files WHERE original_name LIKE ?", (base_name + "%",)
    ).fetchall()
    if existing:
        base_name = f"{base_name} ({len(existing) + 1})"
    conn.execute(
        "INSERT INTO files (id, original_name, mp3_path, duration, transcription, created_at, notebook) "
        "VALUES (?, ?, NULL, 0, '', ?, ?)",
        (file_id, base_name, datetime.now().isoformat(), notebook)
    )
    conn.commit()
    conn.close()
    return jsonify({"file_id": file_id, "original_name": base_name})


@app.route("/library/quick-capture", methods=["POST"])
def quick_capture():
    """Quick capture: create a note in Inbox with optional content."""
    data = request.get_json() or {}
    content = data.get("content", "")
    title = data.get("title", "").strip()
    file_id = str(uuid.uuid4())
    now = datetime.now()
    if not title:
        title = f"Quick Note {now.strftime('%Y-%m-%d %H:%M')}"
    conn = get_db()
    conn.execute(
        "INSERT INTO files (id, original_name, mp3_path, duration, transcription, created_at, notebook) "
        "VALUES (?, ?, NULL, 0, ?, ?, 'Inbox')",
        (file_id, title, content, now.isoformat())
    )
    conn.commit()
    conn.close()
    return jsonify({"file_id": file_id, "original_name": title})


@app.route("/library/<file_id>")
def library_get(file_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "File not found"}), 404
    return jsonify(dict(row))


@app.route("/library/<file_id>/text", methods=["PUT"])
def library_update_text(file_id):
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    conn = get_db()
    # Conflict protection: check if note was modified since client last loaded it
    expected_updated = data.get("expected_updated_at")
    if expected_updated:
        row = conn.execute("SELECT updated_at FROM files WHERE id = ?", (file_id,)).fetchone()
        if row and row["updated_at"] and row["updated_at"] != expected_updated:
            conn.close()
            return jsonify({"error": "conflict", "server_updated_at": row["updated_at"]}), 409
    # Save version snapshot before overwriting
    old = conn.execute("SELECT original_name, transcription, transcription_en FROM files WHERE id = ?", (file_id,)).fetchone()
    if old and (old["transcription"] or old["transcription_en"]):
        # Only save version if content actually changed
        old_text = old["transcription"] or ""
        old_text_en = old["transcription_en"] or ""
        new_text = data.get("text", old_text)
        new_text_en = data.get("text_en", old_text_en)
        if old_text != new_text or old_text_en != new_text_en:
            conn.execute(
                "INSERT INTO note_versions (id, file_id, title, transcription, transcription_en, created_at, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), file_id, old["original_name"], old_text, old_text_en,
                 datetime.now().isoformat(), "autosave")
            )
            # Keep only last 50 versions per note
            conn.execute(
                "DELETE FROM note_versions WHERE file_id = ? AND id NOT IN "
                "(SELECT id FROM note_versions WHERE file_id = ? ORDER BY created_at DESC LIMIT 50)",
                (file_id, file_id)
            )
    if "text" in data:
        conn.execute("UPDATE files SET transcription = ? WHERE id = ?", (data["text"], file_id))
    if "text_en" in data:
        conn.execute("UPDATE files SET transcription_en = ? WHERE id = ?", (data["text_en"], file_id))
    now = datetime.now().isoformat()
    conn.execute("UPDATE files SET updated_at = ? WHERE id = ?", (now, file_id))
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "updated_at": now})


@app.route("/library/<file_id>/rename", methods=["PUT"])
def rename_note(file_id):
    data = request.get_json()
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name required"}), 400
    conn = get_db()
    conn.execute("UPDATE files SET original_name = ? WHERE id = ?", (name, file_id))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/library/<file_id>/notebook", methods=["PUT"])
def update_note_notebook(file_id):
    data = request.get_json()
    notebook = data.get("notebook") or DEFAULT_NOTEBOOK
    conn = get_db()
    conn.execute("UPDATE files SET notebook = ? WHERE id = ?", (notebook, file_id))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/library/<file_id>/star", methods=["PUT"])
def toggle_star(file_id):
    conn = get_db()
    row = conn.execute("SELECT starred FROM files WHERE id = ?", (file_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "File not found"}), 404
    new_val = 0 if row["starred"] else 1
    conn.execute("UPDATE files SET starred = ? WHERE id = ?", (new_val, file_id))
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "starred": new_val})


@app.route("/library/<file_id>/duplicate", methods=["POST"])
def duplicate_note(file_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "File not found"}), 404

    new_id = str(uuid.uuid4())
    new_name = row["original_name"] + " (copy)"
    now = datetime.now().isoformat()

    # Copy MP3 file if audio note
    new_mp3 = None
    if row["mp3_path"]:
        new_mp3 = new_id + ".mp3"
        src = os.path.join(LIBRARY_FOLDER, row["mp3_path"])
        dst = os.path.join(LIBRARY_FOLDER, new_mp3)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    conn.execute(
        "INSERT INTO files (id, original_name, mp3_path, duration, transcription, transcription_en, "
        "created_at, updated_at, tags, notebook, starred) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
        (new_id, new_name, new_mp3, row["duration"], row["transcription"], row["transcription_en"],
         now, now, row["tags"], row["notebook"])
    )
    conn.commit()
    conn.close()
    return jsonify({"file_id": new_id, "original_name": new_name})


@app.route("/library/<file_id>/trash", methods=["PUT"])
def trash_note(file_id):
    conn = get_db()
    conn.execute("UPDATE files SET trashed_at = ? WHERE id = ?", (datetime.now().isoformat(), file_id))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/library/<file_id>/restore", methods=["PUT"])
def restore_note(file_id):
    conn = get_db()
    conn.execute("UPDATE files SET trashed_at = NULL WHERE id = ?", (file_id,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/library/empty-trash", methods=["POST"])
def empty_trash():
    conn = get_db()
    trashed = conn.execute("SELECT id, mp3_path FROM files WHERE trashed_at IS NOT NULL").fetchall()
    for row in trashed:
        if row["mp3_path"]:
            mp3_full = os.path.join(LIBRARY_FOLDER, row["mp3_path"])
            if os.path.exists(mp3_full):
                os.remove(mp3_full)
    conn.execute("DELETE FROM files WHERE trashed_at IS NOT NULL")
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/library/<file_id>/translate", methods=["POST"])
def library_translate(file_id):
    """Re-translate a file's transcription to English."""
    conn = get_db()
    row = conn.execute("SELECT transcription FROM files WHERE id = ?", (file_id,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "File not found"}), 404
    text_en = translate_to_english(row["transcription"])
    if not text_en:
        return jsonify({"error": "OPENAI_API_KEY not set"}), 500
    conn = get_db()
    conn.execute("UPDATE files SET transcription_en = ? WHERE id = ?", (text_en, file_id))
    conn.commit()
    conn.close()
    return jsonify({"text_en": text_en})


@app.route("/library/<file_id>", methods=["DELETE"])
def library_delete(file_id):
    """Permanent delete (used from trash) or move to trash."""
    conn = get_db()
    row = conn.execute("SELECT mp3_path, trashed_at FROM files WHERE id = ?", (file_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "File not found"}), 404

    if row["trashed_at"]:
        # Already in trash — permanent delete
        if row["mp3_path"]:
            mp3_full = os.path.join(LIBRARY_FOLDER, row["mp3_path"])
            if os.path.exists(mp3_full):
                os.remove(mp3_full)
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
    else:
        # Move to trash
        conn.execute("UPDATE files SET trashed_at = ? WHERE id = ?", (datetime.now().isoformat(), file_id))

    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/library/<file_id>/audio")
def library_audio(file_id):
    conn = get_db()
    row = conn.execute("SELECT mp3_path FROM files WHERE id = ?", (file_id,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "File not found"}), 404
    return send_file(os.path.join(LIBRARY_FOLDER, row["mp3_path"]), mimetype="audio/mpeg")


@app.route("/topic-map")
def get_topic_map():
    conn = get_db()
    cached = conn.execute(
        "SELECT value, file_hash, updated_at FROM cache WHERE key = 'topic_map'"
    ).fetchone()
    if not cached or not cached["value"]:
        conn.close()
        return jsonify({"exists": False})
    current_hash = compute_library_hash(conn)
    conn.close()
    return jsonify({
        "exists": True,
        "stale": cached["file_hash"] != current_hash,
        "updated_at": cached["updated_at"],
        "data": json.loads(cached["value"]),
    })


@app.route("/build-topic-map", methods=["POST"])
def build_topic_map():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY not set."}), 500

    conn = get_db()

    # Check cache
    current_hash = compute_library_hash(conn)
    cached = conn.execute(
        "SELECT value, file_hash FROM cache WHERE key = 'topic_map'"
    ).fetchone()
    if cached and cached["file_hash"] == current_hash:
        conn.close()
        return jsonify(json.loads(cached["value"]))

    # Gather all transcriptions
    rows = conn.execute(
        "SELECT id, original_name, transcription_en, transcription FROM files "
        "WHERE transcription IS NOT NULL ORDER BY created_at"
    ).fetchall()

    if len(rows) < 2:
        conn.close()
        return jsonify({"error": "Need at least 2 transcribed files to build a topic map."}), 400

    MAX_CHARS_PER_FILE = 2000
    file_entries = []
    file_lookup = {}
    for r in rows:
        text_en = r["transcription_en"] or ""
        text_orig = r["transcription"] or ""
        if not text_en and not text_orig:
            continue
        # Include both languages so GPT can return keywords for each
        if text_en and text_orig and text_en.strip() != text_orig.strip():
            half = MAX_CHARS_PER_FILE // 2
            truncated = f"[English]:\n{text_en[:half]}\n[Original]:\n{text_orig[:half]}"
        else:
            text = text_en or text_orig
            truncated = text[:MAX_CHARS_PER_FILE]
            if len(text) > MAX_CHARS_PER_FILE:
                truncated += "..."
        file_entries.append(
            f'[File ID: {r["id"]}, Name: "{r["original_name"]}"]\n{truncated}'
        )
        file_lookup[r["id"]] = r["original_name"]

    combined_text = "\n\n---\n\n".join(file_entries)

    # Safety: limit total prompt size
    MAX_TOTAL_CHARS = 80000
    if len(combined_text) > MAX_TOTAL_CHARS:
        chars_per = MAX_TOTAL_CHARS // len(rows)
        file_entries = []
        for r in rows:
            text_en = r["transcription_en"] or ""
            text_orig = r["transcription"] or ""
            if text_en and text_orig and text_en.strip() != text_orig.strip():
                half = chars_per // 2
                text = f"[English]:\n{text_en[:half]}\n[Original]:\n{text_orig[:half]}"
            else:
                text = (text_en or text_orig)[:chars_per]
            file_entries.append(f'[File ID: {r["id"]}, Name: "{r["original_name"]}"]\n{text}')
        combined_text = "\n\n---\n\n".join(file_entries)

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are analyzing multiple speech transcriptions from the same person's library. "
                    "Identify recurring TOPICS across these files and find connections between them.\n\n"
                    "Return a JSON object with exactly this structure:\n"
                    '{"topics": [{"id": 1, "label": "Topic Name", "file_ids": ["uuid1", "uuid2"], "keywords": ["keyword1", "phrase two"]}], '
                    '"edges": [{"from": 1, "to": 2, "label": "shared characteristic"}]}\n\n'
                    "Rules:\n"
                    "- Topics should be high-level themes, projects, or subjects that appear across files\n"
                    "- Each topic MUST list the file_ids (UUIDs) of files where that topic appears\n"
                    "- Each topic MUST include a 'keywords' array of 3-8 short phrases (1-3 words each) that appear VERBATIM in the source transcriptions\n"
                    "- If both [English] and [Original] texts are provided, include keywords from BOTH languages so the topic can be found in either version\n"
                    "- Keywords should be specific words or phrases from the text that identify where the topic is discussed\n"
                    "- Only use file IDs that appear in the input (the UUID after 'File ID:')\n"
                    "- Extract 5-20 topics depending on content volume\n"
                    "- Topic labels: concise but descriptive (2-6 words)\n"
                    "- Edges connect topics that share thematic relationships\n"
                    "- Edge labels: describe the relationship ('both involve', 'enables', 'contrasts with', etc.)\n"
                    "- Aim for meaningful connections, not obvious ones\n"
                    "- Use the same language as the input text\n"
                    "- Return valid JSON only"
                ),
            },
            {
                "role": "user",
                "content": f"Identify cross-file topics and their connections:\n\n{combined_text}",
            },
        ],
    )

    try:
        topic_map = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, IndexError):
        conn.close()
        return jsonify({"error": "Failed to parse topic map from GPT response"}), 500

    topic_map["file_lookup"] = file_lookup

    # Cache result
    conn.execute(
        "INSERT OR REPLACE INTO cache (key, value, file_hash, updated_at) VALUES (?, ?, ?, ?)",
        ("topic_map", json.dumps(topic_map), current_hash, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

    return jsonify(topic_map)


# --- Tags ---

@app.route("/tags")
def list_tags():
    """List all unique tags with note counts."""
    conn = get_db()
    rows = conn.execute(
        "SELECT tags FROM files WHERE tags IS NOT NULL AND tags != '' AND trashed_at IS NULL"
    ).fetchall()
    conn.close()
    tag_counts = {}
    for r in rows:
        try:
            for t in json.loads(r["tags"]):
                tag_counts[t] = tag_counts.get(t, 0) + 1
        except (json.JSONDecodeError, TypeError):
            pass
    result = [{"name": k, "count": v} for k, v in sorted(tag_counts.items())]
    return jsonify(result)


@app.route("/tags/rename", methods=["PUT"])
def rename_tag():
    data = request.get_json()
    old_name = (data.get("old") or "").strip()
    new_name = (data.get("new") or "").strip()
    if not old_name or not new_name:
        return jsonify({"error": "Both old and new tag names required"}), 400
    conn = get_db()
    rows = conn.execute(
        "SELECT id, tags FROM files WHERE tags LIKE ? AND trashed_at IS NULL",
        (f'%"{old_name}"%',)
    ).fetchall()
    updated = 0
    for r in rows:
        try:
            tags = json.loads(r["tags"])
            if old_name in tags:
                tags = [new_name if t == old_name else t for t in tags]
                # Deduplicate
                tags = list(dict.fromkeys(tags))
                conn.execute("UPDATE files SET tags = ? WHERE id = ?", (json.dumps(tags), r["id"]))
                updated += 1
        except (json.JSONDecodeError, TypeError):
            pass
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "updated": updated})


@app.route("/tags/merge", methods=["POST"])
def merge_tags():
    data = request.get_json()
    sources = data.get("sources", [])
    target = (data.get("target") or "").strip()
    if not sources or not target:
        return jsonify({"error": "Sources and target required"}), 400
    conn = get_db()
    rows = conn.execute(
        "SELECT id, tags FROM files WHERE tags IS NOT NULL AND trashed_at IS NULL"
    ).fetchall()
    updated = 0
    for r in rows:
        try:
            tags = json.loads(r["tags"])
            original = list(tags)
            tags = [target if t in sources else t for t in tags]
            tags = list(dict.fromkeys(tags))
            if tags != original:
                conn.execute("UPDATE files SET tags = ? WHERE id = ?", (json.dumps(tags), r["id"]))
                updated += 1
        except (json.JSONDecodeError, TypeError):
            pass
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "updated": updated})


@app.route("/library/<file_id>/tags", methods=["PUT"])
def update_note_tags(file_id):
    data = request.get_json()
    tags = data.get("tags", [])
    conn = get_db()
    conn.execute("UPDATE files SET tags = ? WHERE id = ?", (json.dumps(tags), file_id))
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "tags": tags})


# --- Notebook Stacks ---

@app.route("/notebooks/<name>/stack", methods=["PUT"])
def set_notebook_stack(name):
    data = request.get_json()
    stack = (data.get("stack") or "").strip()
    conn = get_db()
    conn.execute("UPDATE notebooks SET stack = ? WHERE name = ?", (stack or None, name))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


# --- Bulk Actions ---

@app.route("/library/bulk", methods=["POST"])
def bulk_action():
    data = request.get_json()
    ids = data.get("ids", [])
    action = data.get("action", "")
    if not ids:
        return jsonify({"error": "No notes selected"}), 400

    conn = get_db()
    placeholders = ",".join("?" * len(ids))

    if action == "trash":
        now = datetime.now().isoformat()
        conn.execute(f"UPDATE files SET trashed_at = ? WHERE id IN ({placeholders})", [now] + ids)
    elif action == "star":
        conn.execute(f"UPDATE files SET starred = 1 WHERE id IN ({placeholders})", ids)
    elif action == "unstar":
        conn.execute(f"UPDATE files SET starred = 0 WHERE id IN ({placeholders})", ids)
    elif action == "move":
        notebook = data.get("notebook", DEFAULT_NOTEBOOK)
        conn.execute(f"UPDATE files SET notebook = ? WHERE id IN ({placeholders})", [notebook] + ids)
    elif action == "tag":
        tag = (data.get("tag") or "").strip()
        if tag:
            rows = conn.execute(f"SELECT id, tags FROM files WHERE id IN ({placeholders})", ids).fetchall()
            for r in rows:
                try:
                    tags = json.loads(r["tags"]) if r["tags"] else []
                except (json.JSONDecodeError, TypeError):
                    tags = []
                if tag not in tags:
                    tags.append(tag)
                conn.execute("UPDATE files SET tags = ? WHERE id = ?", (json.dumps(tags), r["id"]))
    elif action == "untag":
        tag = (data.get("tag") or "").strip()
        if tag:
            rows = conn.execute(f"SELECT id, tags FROM files WHERE id IN ({placeholders})", ids).fetchall()
            for r in rows:
                try:
                    tags = json.loads(r["tags"]) if r["tags"] else []
                except (json.JSONDecodeError, TypeError):
                    tags = []
                tags = [t for t in tags if t != tag]
                conn.execute("UPDATE files SET tags = ? WHERE id = ?", (json.dumps(tags), r["id"]))
    elif action == "delete":
        # Permanent delete for trashed notes
        for fid in ids:
            row = conn.execute("SELECT mp3_path FROM files WHERE id = ?", (fid,)).fetchone()
            if row and row["mp3_path"]:
                mp3_full = os.path.join(LIBRARY_FOLDER, row["mp3_path"])
                if os.path.exists(mp3_full):
                    os.remove(mp3_full)
        conn.execute(f"DELETE FROM files WHERE id IN ({placeholders})", ids)
    else:
        conn.close()
        return jsonify({"error": f"Unknown action: {action}"}), 400

    conn.commit()
    conn.close()
    return jsonify({"ok": True, "count": len(ids)})


# --- Saved Searches ---

@app.route("/saved-searches")
def list_saved_searches():
    conn = get_db()
    rows = conn.execute("SELECT * FROM saved_searches ORDER BY created_at DESC").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/saved-searches", methods=["POST"])
def create_saved_search():
    data = request.get_json()
    name = (data.get("name") or "").strip()
    query = (data.get("query") or "").strip()
    filters = data.get("filters", {})
    if not name:
        return jsonify({"error": "Name required"}), 400
    search_id = str(uuid.uuid4())
    conn = get_db()
    conn.execute(
        "INSERT INTO saved_searches (id, name, query, filters, created_at) VALUES (?, ?, ?, ?, ?)",
        (search_id, name, query, json.dumps(filters), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return jsonify({"id": search_id, "name": name, "query": query, "filters": filters})


@app.route("/saved-searches/<search_id>", methods=["DELETE"])
def delete_saved_search(search_id):
    conn = get_db()
    conn.execute("DELETE FROM saved_searches WHERE id = ?", (search_id,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


# --- Attachments ---

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "svg", "webp"}
ATTACH_EXTENSIONS = IMAGE_EXTENSIONS | {"pdf", "doc", "docx", "xls", "xlsx", "zip", "mp3", "wav", "m4a"}


@app.route("/library/<file_id>/attachments", methods=["POST"])
def upload_attachment(file_id):
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    att_id = str(uuid.uuid4())
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    save_name = att_id + ("." + ext if ext else "")
    save_path = os.path.join(ATTACHMENTS_FOLDER, save_name)
    file.save(save_path)

    mime = mimetypes.guess_type(file.filename)[0] or "application/octet-stream"
    size = os.path.getsize(save_path)

    conn = get_db()
    conn.execute(
        "INSERT INTO attachments (id, file_id, filename, mime_type, size, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (att_id, file_id, file.filename, mime, size, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

    is_image = ext in IMAGE_EXTENSIONS
    return jsonify({
        "id": att_id, "filename": file.filename, "mime_type": mime,
        "size": size, "is_image": is_image,
        "url": f"/attachments/{att_id}"
    })


@app.route("/attachments/<att_id>")
def serve_attachment(att_id):
    conn = get_db()
    row = conn.execute("SELECT filename, mime_type FROM attachments WHERE id = ?", (att_id,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Not found"}), 404

    # Find the file (with any extension)
    for f in os.listdir(ATTACHMENTS_FOLDER):
        if f.startswith(att_id):
            return send_file(
                os.path.join(ATTACHMENTS_FOLDER, f),
                mimetype=row["mime_type"],
                download_name=row["filename"]
            )
    return jsonify({"error": "File missing"}), 404


@app.route("/attachments/<att_id>", methods=["DELETE"])
def delete_attachment(att_id):
    conn = get_db()
    row = conn.execute("SELECT id FROM attachments WHERE id = ?", (att_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Not found"}), 404
    conn.execute("DELETE FROM attachments WHERE id = ?", (att_id,))
    conn.commit()
    conn.close()
    # Remove file from disk
    for f in os.listdir(ATTACHMENTS_FOLDER):
        if f.startswith(att_id):
            os.remove(os.path.join(ATTACHMENTS_FOLDER, f))
            break
    return jsonify({"ok": True})


@app.route("/library/<file_id>/attachments")
def list_attachments(file_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, filename, mime_type, size, created_at FROM attachments WHERE file_id = ? ORDER BY created_at",
        (file_id,)
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["url"] = f"/attachments/{r['id']}"
        d["is_image"] = any(r["filename"].lower().endswith("." + e) for e in IMAGE_EXTENSIONS)
        result.append(d)
    return jsonify(result)


@app.route("/files")
def list_all_files():
    """List all attachments across all notes."""
    conn = get_db()
    q = request.args.get("q", "").strip()
    ftype = request.args.get("type", "")  # "media", "docs", or ""
    query = (
        "SELECT a.id, a.filename, a.mime_type, a.size, a.created_at, a.file_id, "
        "f.original_name as note_name, f.notebook "
        "FROM attachments a LEFT JOIN files f ON a.file_id = f.id "
        "WHERE f.trashed_at IS NULL"
    )
    params = []
    if q:
        query += " AND a.filename LIKE ?"
        params.append(f"%{q}%")
    if ftype == "media":
        query += " AND (a.mime_type LIKE 'image/%' OR a.mime_type LIKE 'audio/%' OR a.mime_type LIKE 'video/%')"
    elif ftype == "docs":
        query += " AND a.mime_type NOT LIKE 'image/%' AND a.mime_type NOT LIKE 'audio/%' AND a.mime_type NOT LIKE 'video/%'"
    query += " ORDER BY a.created_at DESC"
    rows = conn.execute(query, params).fetchall()
    total_size = conn.execute(
        "SELECT COALESCE(SUM(a.size), 0) FROM attachments a LEFT JOIN files f ON a.file_id = f.id WHERE f.trashed_at IS NULL"
    ).fetchone()[0]
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["url"] = f"/attachments/{r['id']}"
        d["is_image"] = any(r["filename"].lower().endswith("." + e) for e in IMAGE_EXTENSIONS)
        result.append(d)
    return jsonify({"files": result, "total_size": total_size})


@app.route("/files/upload", methods=["POST"])
def upload_standalone_file():
    """Upload a file not attached to any specific note — creates a note for it."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    # Create a note to hold this file
    file_id = str(uuid.uuid4())
    conn = get_db()
    conn.execute(
        "INSERT INTO files (id, original_name, mp3_path, duration, transcription, created_at, notebook) "
        "VALUES (?, ?, NULL, 0, '', ?, ?)",
        (file_id, file.filename, datetime.now().isoformat(), "My Notebook")
    )

    # Save attachment
    att_id = str(uuid.uuid4())
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    save_name = att_id + ("." + ext if ext else "")
    save_path = os.path.join(ATTACHMENTS_FOLDER, save_name)
    file.save(save_path)
    mime = mimetypes.guess_type(file.filename)[0] or "application/octet-stream"
    size = os.path.getsize(save_path)
    conn.execute(
        "INSERT INTO attachments (id, file_id, filename, mime_type, size, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (att_id, file_id, file.filename, mime, size, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "file_id": file_id, "att_id": att_id})


@app.route("/library/search-titles")
def search_note_titles():
    """Search note titles for internal linking."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    conn = get_db()
    rows = conn.execute(
        "SELECT id, original_name FROM files WHERE trashed_at IS NULL AND original_name LIKE ? LIMIT 10",
        (f"%{q}%",)
    ).fetchall()
    conn.close()
    return jsonify([{"id": r["id"], "name": r["original_name"]} for r in rows])


# --- Evernote Import ---

def parse_enex_date(date_str):
    """Parse Evernote date format (20231015T123456Z) to ISO format."""
    if not date_str:
        return datetime.now().isoformat()
    try:
        dt = datetime.strptime(date_str.strip(), "%Y%m%dT%H%M%SZ")
        return dt.isoformat()
    except ValueError:
        return datetime.now().isoformat()


def enml_to_html(enml_content):
    """Convert ENML content to clean HTML."""
    if not enml_content:
        return ""
    # Remove XML declaration and DOCTYPE
    content = re.sub(r'<\?xml[^?]*\?>', '', enml_content)
    content = re.sub(r'<!DOCTYPE[^>]*>', '', content)
    # Replace en-note with div
    content = re.sub(r'<en-note[^>]*>', '', content)
    content = re.sub(r'</en-note>', '', content)
    # Remove en-media tags (embedded resources) — keep a placeholder
    content = re.sub(r'<en-media[^>]*/>', '', content)
    content = re.sub(r'<en-media[^>]*>.*?</en-media>', '', content, flags=re.DOTALL)
    # Remove en-crypt, en-todo (checkboxes become text)
    content = re.sub(r'<en-todo\s+checked="true"\s*/>', '[x] ', content)
    content = re.sub(r'<en-todo[^>]*/>', '[ ] ', content)
    content = re.sub(r'<en-crypt[^>]*>.*?</en-crypt>', '', content, flags=re.DOTALL)
    return content.strip()


@app.route("/import-enex", methods=["POST"])
def import_enex():
    """Import notes from an Evernote .enex export file."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    original_filename = file.filename or "import.enex"
    if not original_filename.lower().endswith(".enex"):
        return jsonify({"error": "Please upload an .enex file (Evernote export)"}), 400

    # Use a safe temp filename for saving, but keep original for notebook name
    safe_name = str(uuid.uuid4()) + ".enex"
    filepath = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(filepath)

    try:
        # Parse with explicit UTF-8 to handle Cyrillic and other non-ASCII metadata
        with open(filepath, "r", encoding="utf-8") as f:
            xml_content = f.read()
        root = ET.fromstring(xml_content)

        notes = root.findall("note")
        if not notes:
            return jsonify({"error": "No notes found in the .enex file"}), 400

        conn = get_db()
        imported = []
        created_notebooks = set()

        # Notebook override from form data, or derive from filename
        user_notebook = request.form.get("notebook", "").strip()
        filename_notebook = user_notebook or os.path.splitext(original_filename)[0].strip()

        for note in notes:
            file_id = str(uuid.uuid4())

            title = note.findtext("title", "Untitled")
            content_el = note.find("content")
            enml_content = content_el.text if content_el is not None else ""
            html_content = enml_to_html(enml_content)

            created = parse_enex_date(note.findtext("created"))
            updated = parse_enex_date(note.findtext("updated"))

            # Collect tags
            tag_elements = note.findall("tag")
            tags = json.dumps([t.text for t in tag_elements if t.text]) if tag_elements else None

            # Extract notebook: try <notebook> element, then note-attributes/notebook,
            # then use the .enex filename (Evernote names exports after the source notebook)
            notebook_name = (
                note.findtext("notebook")
                or note.findtext("note-attributes/notebook")
                or filename_notebook
                or DEFAULT_NOTEBOOK
            ).strip()
            if notebook_name and notebook_name not in created_notebooks:
                conn.execute(
                    "INSERT OR IGNORE INTO notebooks (name, created_at) VALUES (?, ?)",
                    (notebook_name, datetime.now().isoformat())
                )
                created_notebooks.add(notebook_name)

            # Extract additional note-attributes metadata
            attrs = note.find("note-attributes")
            source_url = attrs.findtext("source-url") if attrs is not None else None
            author = attrs.findtext("author") if attrs is not None else None
            latitude = attrs.findtext("latitude") if attrs is not None else None
            longitude = attrs.findtext("longitude") if attrs is not None else None

            # Append source metadata to content if present
            meta_html = ""
            if source_url:
                meta_html += f'<p><small>Source: <a href="{source_url}">{source_url}</a></small></p>'
            if author:
                meta_html += f'<p><small>Author: {author}</small></p>'
            if latitude and longitude:
                meta_html += f'<p><small>Location: {latitude}, {longitude}</small></p>'
            if meta_html:
                html_content = html_content + '<hr>' + meta_html

            conn.execute(
                "INSERT INTO files (id, original_name, mp3_path, duration, transcription, created_at, updated_at, tags, notebook) "
                "VALUES (?, ?, NULL, 0, ?, ?, ?, ?, ?)",
                (file_id, title, html_content, created, updated, tags, notebook_name)
            )
            imported.append({"file_id": file_id, "title": title, "notebook": notebook_name})

        conn.commit()
        conn.close()

        return jsonify({
            "imported": len(imported),
            "notes": imported
        })

    except ET.ParseError as e:
        return jsonify({"error": f"Invalid XML: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# --- Version History ---

@app.route("/library/<file_id>/versions")
def list_versions(file_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, title, created_at, source, "
        "length(COALESCE(transcription, '')) + length(COALESCE(transcription_en, '')) as size "
        "FROM note_versions WHERE file_id = ? ORDER BY created_at DESC",
        (file_id,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/library/<file_id>/versions/<version_id>")
def get_version(file_id, version_id):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM note_versions WHERE id = ? AND file_id = ?",
        (version_id, file_id)
    ).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Version not found"}), 404
    return jsonify(dict(row))


@app.route("/library/<file_id>/versions/<version_id>/restore", methods=["POST"])
def restore_version(file_id, version_id):
    conn = get_db()
    ver = conn.execute(
        "SELECT * FROM note_versions WHERE id = ? AND file_id = ?",
        (version_id, file_id)
    ).fetchone()
    if not ver:
        conn.close()
        return jsonify({"error": "Version not found"}), 404
    # Save current state as a version before restoring
    old = conn.execute("SELECT original_name, transcription, transcription_en FROM files WHERE id = ?", (file_id,)).fetchone()
    if old:
        conn.execute(
            "INSERT INTO note_versions (id, file_id, title, transcription, transcription_en, created_at, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), file_id, old["original_name"],
             old["transcription"] or "", old["transcription_en"] or "",
             datetime.now().isoformat(), "before_restore")
        )
    # Restore
    conn.execute(
        "UPDATE files SET transcription = ?, transcription_en = ?, updated_at = ? WHERE id = ?",
        (ver["transcription"], ver["transcription_en"], datetime.now().isoformat(), file_id)
    )
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


# --- Export ---

@app.route("/library/<file_id>/export/<fmt>")
def export_note(file_id, fmt):
    conn = get_db()
    row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Not found"}), 404
    note = dict(row)
    title = note["original_name"] or "Untitled"
    safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')

    if fmt == "md":
        content = f"# {title}\n\n"
        text_en = note.get("transcription_en") or ""
        text_orig = note.get("transcription") or ""
        if text_en:
            plain = html_to_markdown(text_en)
            content += plain + "\n"
        if text_orig and text_orig != text_en:
            content += "\n---\n\n## Original\n\n" + html_to_markdown(text_orig) + "\n"
        if note.get("tags"):
            try:
                tags = json.loads(note["tags"])
                if tags:
                    content += "\n---\nTags: " + ", ".join(tags) + "\n"
            except (json.JSONDecodeError, TypeError):
                pass
        return Response(content, mimetype="text/markdown",
                        headers={"Content-Disposition": f'attachment; filename="{safe_title}.md"'})

    elif fmt == "html":
        text_en = note.get("transcription_en") or ""
        text_orig = note.get("transcription") or ""
        html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title>"
        html += "<style>body{font-family:sans-serif;max-width:800px;margin:40px auto;padding:0 20px;line-height:1.6}"
        html += "h1{border-bottom:2px solid #eee;padding-bottom:10px}</style></head><body>"
        html += f"<h1>{title}</h1>"
        if text_en:
            html += text_en
        if text_orig and text_orig != text_en:
            html += "<hr><h2>Original</h2>" + text_orig
        tags_str = ""
        if note.get("tags"):
            try:
                tags = json.loads(note["tags"])
                if tags:
                    tags_str = "<hr><p><strong>Tags:</strong> " + ", ".join(tags) + "</p>"
            except (json.JSONDecodeError, TypeError):
                pass
        html += tags_str
        html += f"<hr><p><small>Created: {note['created_at']}"
        if note.get("notebook"):
            html += f" | Notebook: {note['notebook']}"
        html += "</small></p></body></html>"
        return Response(html, mimetype="text/html",
                        headers={"Content-Disposition": f'attachment; filename="{safe_title}.html"'})

    elif fmt == "json":
        export = {
            "id": note["id"],
            "title": title,
            "notebook": note.get("notebook"),
            "tags": json.loads(note["tags"]) if note.get("tags") else [],
            "created_at": note["created_at"],
            "updated_at": note.get("updated_at"),
            "transcription": note.get("transcription"),
            "transcription_en": note.get("transcription_en"),
            "duration": note.get("duration"),
            "starred": bool(note.get("starred")),
        }
        return Response(json.dumps(export, indent=2, ensure_ascii=False), mimetype="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{safe_title}.json"'})

    elif fmt == "enex":
        from xml.sax.saxutils import escape as xml_escape
        created = note["created_at"].replace("-", "").replace(":", "").replace("T", "T")[:15] + "Z"
        text_en = note.get("transcription_en") or note.get("transcription") or ""
        enex = '<?xml version="1.0" encoding="UTF-8"?>\n'
        enex += '<!DOCTYPE en-export SYSTEM "http://xml.evernote.com/pub/evernote-export4.dtd">\n'
        enex += '<en-export>\n<note>\n'
        enex += f'  <title>{xml_escape(title)}</title>\n'
        enex += f'  <content><![CDATA[<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        enex += '<!DOCTYPE en-note SYSTEM "http://xml.evernote.com/pub/enml2.dtd">\n'
        enex += f'<en-note>{text_en}</en-note>]]></content>\n'
        enex += f'  <created>{created}</created>\n'
        if note.get("tags"):
            try:
                tags = json.loads(note["tags"])
                for t in tags:
                    enex += f'  <tag>{xml_escape(t)}</tag>\n'
            except (json.JSONDecodeError, TypeError):
                pass
        enex += '</note>\n</en-export>'
        return Response(enex, mimetype="application/xml",
                        headers={"Content-Disposition": f'attachment; filename="{safe_title}.enex"'})

    return jsonify({"error": "Unsupported format. Use: md, html, json, enex"}), 400


@app.route("/library/export-all/<fmt>")
def export_all_notes(fmt):
    """Export all non-trashed notes as a single file."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM files WHERE trashed_at IS NULL ORDER BY created_at DESC"
    ).fetchall()
    conn.close()

    if fmt == "json":
        notes = []
        for r in rows:
            note = dict(r)
            notes.append({
                "id": note["id"],
                "title": note["original_name"],
                "notebook": note.get("notebook"),
                "tags": json.loads(note["tags"]) if note.get("tags") else [],
                "created_at": note["created_at"],
                "updated_at": note.get("updated_at"),
                "transcription": note.get("transcription"),
                "transcription_en": note.get("transcription_en"),
                "duration": note.get("duration"),
                "starred": bool(note.get("starred")),
            })
        return Response(json.dumps(notes, indent=2, ensure_ascii=False), mimetype="application/json",
                        headers={"Content-Disposition": 'attachment; filename="sememes_export.json"'})

    elif fmt == "enex":
        from xml.sax.saxutils import escape as xml_escape
        enex = '<?xml version="1.0" encoding="UTF-8"?>\n'
        enex += '<!DOCTYPE en-export SYSTEM "http://xml.evernote.com/pub/evernote-export4.dtd">\n'
        enex += '<en-export>\n'
        for r in rows:
            note = dict(r)
            title = note["original_name"] or "Untitled"
            created = note["created_at"].replace("-", "").replace(":", "").replace("T", "T")[:15] + "Z"
            text = note.get("transcription_en") or note.get("transcription") or ""
            enex += '<note>\n'
            enex += f'  <title>{xml_escape(title)}</title>\n'
            enex += f'  <content><![CDATA[<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
            enex += '<!DOCTYPE en-note SYSTEM "http://xml.evernote.com/pub/enml2.dtd">\n'
            enex += f'<en-note>{text}</en-note>]]></content>\n'
            enex += f'  <created>{created}</created>\n'
            if note.get("tags"):
                try:
                    tags = json.loads(note["tags"])
                    for t in tags:
                        enex += f'  <tag>{xml_escape(t)}</tag>\n'
                except (json.JSONDecodeError, TypeError):
                    pass
            enex += '</note>\n'
        enex += '</en-export>'
        return Response(enex, mimetype="application/xml",
                        headers={"Content-Disposition": 'attachment; filename="sememes_export.enex"'})

    return jsonify({"error": "Bulk export supports: json, enex"}), 400


def html_to_markdown(html_text):
    """Simple HTML to Markdown converter for export."""
    text = html_text
    text = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1\n', text, flags=re.DOTALL)
    text = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1\n', text, flags=re.DOTALL)
    text = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1\n', text, flags=re.DOTALL)
    text = re.sub(r'<strong>(.*?)</strong>', r'**\1**', text, flags=re.DOTALL)
    text = re.sub(r'<b>(.*?)</b>', r'**\1**', text, flags=re.DOTALL)
    text = re.sub(r'<em>(.*?)</em>', r'*\1*', text, flags=re.DOTALL)
    text = re.sub(r'<i>(.*?)</i>', r'*\1*', text, flags=re.DOTALL)
    text = re.sub(r'<s>(.*?)</s>', r'~~\1~~', text, flags=re.DOTALL)
    text = re.sub(r'<a\s+href="([^"]*)"[^>]*>(.*?)</a>', r'[\2](\1)', text, flags=re.DOTALL)
    text = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1\n', text, flags=re.DOTALL)
    text = re.sub(r'<blockquote[^>]*>(.*?)</blockquote>', lambda m: '> ' + m.group(1).strip() + '\n', text, flags=re.DOTALL)
    text = re.sub(r'<code>(.*?)</code>', r'`\1`', text, flags=re.DOTALL)
    text = re.sub(r'<pre[^>]*>(.*?)</pre>', r'```\n\1\n```\n', text, flags=re.DOTALL)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<hr\s*/?>', '\n---\n', text)
    text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', text, flags=re.DOTALL)
    text = re.sub(r'<div[^>]*>(.*?)</div>', r'\1\n', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# --- Helpers ---

def sse_event(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def format_duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
