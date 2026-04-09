"""Authentication — registration, login, logout."""

import uuid
from datetime import datetime

from flask import Blueprint, request, redirect, url_for, render_template, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from db import get_db

auth_bp = Blueprint("auth", __name__)
login_manager = LoginManager()
login_manager.login_view = "auth.login"

DEFAULT_NOTEBOOK = "My Notebook"


class User(UserMixin):
    def __init__(self, id, email, display_name, is_admin=False):
        self.id = id
        self.email = email
        self.display_name = display_name
        self.is_admin = is_admin


@login_manager.user_loader
def load_user(user_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if row:
        return User(row["id"], row["email"], row["display_name"], bool(row["is_admin"]))
    return None


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "GET":
        # Check if any users exist — if not, show register page
        conn = get_db()
        count = conn.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"]
        conn.close()
        if count == 0:
            return redirect(url_for("auth.register"))
        return render_template("login.html")

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    remember = request.form.get("remember") == "on"

    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()

    if not row or not check_password_hash(row["password_hash"], password):
        return render_template("login.html", error="Invalid email or password")

    user = User(row["id"], row["email"], row["display_name"], bool(row["is_admin"]))
    login_user(user, remember=remember)

    # Update last_login
    conn = get_db()
    conn.execute("UPDATE users SET last_login = ? WHERE id = ?",
                 (datetime.now().isoformat(), user.id))
    conn.commit()
    conn.close()

    return redirect(url_for("index"))


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "GET":
        return render_template("register.html")

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    confirm = request.form.get("confirm", "")
    display_name = request.form.get("display_name", "").strip()

    if not email or not password:
        return render_template("register.html", error="Email and password are required")
    if password != confirm:
        return render_template("register.html", error="Passwords do not match")
    if len(password) < 6:
        return render_template("register.html", error="Password must be at least 6 characters")

    conn = get_db()
    existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    if existing:
        conn.close()
        return render_template("register.html", error="Email already registered")

    # Check if this is the first user — make them admin
    count = conn.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"]
    is_admin = 1 if count == 0 else 0

    user_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    conn.execute(
        "INSERT INTO users (id, email, password_hash, display_name, is_admin, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, email, generate_password_hash(password), display_name or email, is_admin, now)
    )

    # Create default notebooks for new user
    conn.execute(
        "INSERT INTO notebooks (id, name, created_at, user_id) VALUES (?, ?, ?, ?)",
        (str(uuid.uuid4()), DEFAULT_NOTEBOOK, now, user_id)
    )
    conn.execute(
        "INSERT INTO notebooks (id, name, created_at, user_id) VALUES (?, ?, ?, ?)",
        (str(uuid.uuid4()), "Inbox", now, user_id)
    )
    conn.commit()
    conn.close()

    user = User(user_id, email, display_name or email, bool(is_admin))
    login_user(user, remember=True)
    return redirect(url_for("index"))


@auth_bp.route("/logout", methods=["POST"])
def logout():
    logout_user()
    return redirect(url_for("auth.login"))
