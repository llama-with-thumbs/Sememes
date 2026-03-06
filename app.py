import hashlib
import json
import os
import sqlite3
import subprocess
import uuid
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

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LIBRARY_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"m4a", "mp3", "wav", "ogg", "flac", "webm", "mp4", "wma"}
TEXT_EXTENSIONS = {"txt", "md", "csv", "json", "log", "rtf"}

CHUNK_SECONDS = 120  # 2-minute chunks for progress tracking


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
                "INSERT INTO files (id, original_name, mp3_path, duration, transcription, created_at) VALUES (?, ?, NULL, 0, ?, ?)",
                (file_id, file.filename, text_content, datetime.now().isoformat())
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
            "INSERT INTO files (id, original_name, mp3_path, duration, created_at) VALUES (?, ?, ?, ?, ?)",
            (file_id, file.filename, mp3_filename, duration, datetime.now().isoformat())
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
    rows = conn.execute(
        "SELECT id, original_name, duration, created_at, transcription IS NOT NULL as has_transcription FROM files ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/library/note", methods=["POST"])
def create_note():
    """Create a text-only note (no audio)."""
    file_id = str(uuid.uuid4())
    base_name = f"Note {datetime.now().strftime('%Y-%m-%d')}"
    conn = get_db()
    # Avoid duplicate names
    existing = conn.execute(
        "SELECT original_name FROM files WHERE original_name LIKE ?", (base_name + "%",)
    ).fetchall()
    if existing:
        base_name = f"{base_name} ({len(existing) + 1})"
    conn.execute(
        "INSERT INTO files (id, original_name, mp3_path, duration, transcription, created_at) VALUES (?, ?, NULL, 0, '', ?)",
        (file_id, base_name, datetime.now().isoformat())
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
    conn = get_db()
    row = conn.execute("SELECT mp3_path FROM files WHERE id = ?", (file_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "File not found"}), 404

    if row["mp3_path"]:
        mp3_full = os.path.join(LIBRARY_FOLDER, row["mp3_path"])
        if os.path.exists(mp3_full):
            os.remove(mp3_full)

    conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
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
