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

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max upload

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
LIBRARY_FOLDER = os.path.join(BASE_DIR, "library")
DB_PATH = os.path.join(BASE_DIR, "library.db")

ATTACHMENTS_FOLDER = os.path.join(BASE_DIR, "attachments")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LIBRARY_FOLDER, exist_ok=True)
os.makedirs(ATTACHMENTS_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"m4a", "mp3", "wav", "ogg", "flac", "webm", "mp4", "wma"}
TEXT_EXTENSIONS = {"txt", "md", "csv", "json", "log", "rtf"}

CHUNK_SECONDS = 120  # 2-minute chunks for progress tracking
DEFAULT_NOTEBOOK = "My Notebook"


# --- Database ---

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
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
    # Ensure default notebook exists and assign orphan notes
    DEFAULT_NOTEBOOK = "My Notebook"
    conn.execute(
        "INSERT OR IGNORE INTO notebooks (name, created_at) VALUES (?, ?)",
        (DEFAULT_NOTEBOOK, datetime.now().isoformat())
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

    if search_q:
        conditions.append("(original_name LIKE ? OR transcription LIKE ? OR transcription_en LIKE ?)")
        s = f"%{search_q}%"
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
        d["preview"] = re.sub(r'<[^>]+>', '', raw).strip()[:150]
        result.append(d)
    return jsonify(result)


@app.route("/notebooks")
def list_notebooks():
    conn = get_db()
    nb_rows = conn.execute(
        "SELECT n.name, n.stack, COALESCE(c.cnt, 0) as note_count FROM notebooks n "
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
    if "text" in data:
        conn.execute("UPDATE files SET transcription = ? WHERE id = ?", (data["text"], file_id))
    if "text_en" in data:
        conn.execute("UPDATE files SET transcription_en = ? WHERE id = ?", (data["text_en"], file_id))
    conn.execute("UPDATE files SET updated_at = ? WHERE id = ?", (datetime.now().isoformat(), file_id))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


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
    if not file.filename.lower().endswith(".enex"):
        return jsonify({"error": "Please upload an .enex file (Evernote export)"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        notes = root.findall("note")
        if not notes:
            return jsonify({"error": "No notes found in the .enex file"}), 400

        conn = get_db()
        imported = []

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

            conn.execute(
                "INSERT INTO files (id, original_name, mp3_path, duration, transcription, created_at, updated_at, tags, notebook) "
                "VALUES (?, ?, NULL, 0, ?, ?, ?, ?, ?)",
                (file_id, title, html_content, created, updated, tags, DEFAULT_NOTEBOOK)
            )
            imported.append({"file_id": file_id, "title": title})

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
    app.run(debug=False, host="127.0.0.1", port=5000)
