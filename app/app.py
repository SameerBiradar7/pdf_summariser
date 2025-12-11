# app/app.py
import os
import time
import subprocess
import tempfile
import re
from pymongo import MongoClient
import datetime
from pathlib import Path
from typing import List, Tuple
from flask import (
    Flask,
    request,
    render_template,
    jsonify,
    send_from_directory,
    redirect,
    url_for,
    session,
)

from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from werkzeug.security import generate_password_hash, check_password_hash
import ollama

# local modules
from app.extract import extract_text
from app.chunking import prepare_chunks
from app.embeddings import (
    embed_chunks,
    build_faiss_index,
    persist_index,
    load_index_and_chunks,
    load_embeddings_if_exists,
)
from app.retrieval import retrieve_top_k, assemble_context

# config / folders
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".pdf", ".txt", ".pptx", ".ppt", ".docx", ".md"}

app = Flask(__name__, template_folder=str(BASE_DIR / "app" / "templates"))
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

# Session secret key (for login sessions)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# -------------------------
# MongoDB (local, offline)
# -------------------------
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "ai_summariser")
MONGO_USERS_COL = os.environ.get("MONGO_USERS_COL", "users")

mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB_NAME]
users_col = mongo_db[MONGO_USERS_COL]


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def hash_password(raw_password: str) -> str:
    """Hash a password using Werkzeug (PBKDF2)."""
    raw = str(raw_password or "")
    return generate_password_hash(raw)

def verify_password(raw_password: str, hashed: str) -> bool:
    """Verify password against stored hash."""
    raw = str(raw_password or "")
    if not hashed:
        return False
    try:
        return check_password_hash(hashed, raw)
    except Exception:
        return False




# -------------------------
# HOME
# -------------------------
@app.route("/", methods=["GET"])
def index():
    # Require login to use the summariser dashboard
    if not session.get("user_id"):
        return redirect(url_for("login_page"))

    models = ["gemma3:1b", "qwen2:1.5b", "llama3.2:latest"]
    return render_template("index.html", models=models, default_model=models[0])


# -------------------------
# AUTH PAGES (UI only)
# -------------------------
@app.route("/signup", methods=["GET"])
def signup_page():
    # if already logged in, go to dashboard
    if session.get("user_id"):
        return redirect(url_for("index"))
    return render_template("signup.html")


@app.route("/login", methods=["GET"])
def login_page():
    # if already logged in, go to dashboard
    if session.get("user_id"):
        return redirect(url_for("index"))
    return render_template("login.html")




# -------------------------
# UPLOAD
# -------------------------
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No filename"}), 400

    filename = secure_filename(file.filename)
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    save_path = UPLOAD_FOLDER / filename
    file.save(save_path)

    # use stem as default prefix (safe filename)
    prefix = save_path.stem.replace(" ", "_")
    model_name = request.form.get("model", "gemma3:1b")

    return jsonify({
        "status": "uploaded",
        "filepath": str(save_path),
        "prefix": prefix,
        "model": model_name
    })

# -------------------------
# AUTH APIs (MongoDB)
# -------------------------
@app.route("/signup", methods=["POST"])
def signup_api():
    """
    JSON body:
    {
      "name": "Full Name",
      "email": "user@example.com",
      "password": "1234"
    }
    """
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = normalize_email(data.get("email"))
    password = str(data.get("password") or "").strip()

    # Basic validations (must match your frontend rules)
    if len(name) < 2:
        return jsonify({"ok": False, "error": "Name must be at least 2 characters long"}), 400

    # simple email check
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return jsonify({"ok": False, "error": "Please enter a valid email address"}), 400

    if not re.fullmatch(r"\d{4}", password):
        return jsonify({"ok": False, "error": "Password must be exactly 4 digits"}), 400

    # Check if email already exists
    existing = users_col.find_one({"email": email})
    if existing:
        return jsonify({"ok": False, "error": "This email is already registered"}), 400

    # Create user
    user_doc = {
        "name": name,
        "email": email,
        "password_hash": hash_password(password),
        "created_at": datetime.datetime.utcnow(),
    }
    res = users_col.insert_one(user_doc)

    # Do NOT auto-login here. Frontend already redirects to /login.
    return jsonify({"ok": True, "message": "Account created successfully"}), 201


@app.route("/login", methods=["POST"])
def login_api():
    """
    JSON body:
    {
      "email": "user@example.com",
      "password": "1234"
    }
    """
    data = request.get_json(silent=True) or {}
    email = normalize_email(data.get("email"))
    password = str(data.get("password") or "").strip()

    # Basic validations
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return jsonify({"ok": False, "error": "Please enter a valid email address"}), 400

    if not re.fullmatch(r"\d{4}", password):
        return jsonify({"ok": False, "error": "Password must be exactly 4 digits"}), 400

    user = users_col.find_one({"email": email})
    if not user:
        return jsonify({"ok": False, "error": "Invalid email or password"}), 401

    if not verify_password(password, user.get("password_hash", "")):
        return jsonify({"ok": False, "error": "Invalid email or password"}), 401

    # Set session
    session["user_id"] = str(user["_id"])
    session["user_email"] = user["email"]
    session["user_name"] = user.get("name", "")

    # Optional token for frontend; not really needed for simple session auth
    token = f"session-{session['user_id']}"

    return jsonify({"ok": True, "message": "Login successful", "token": token}), 200


@app.route("/logout", methods=["POST", "GET"])
def logout():
    session.clear()
    return redirect(url_for("login_page"))




# -------------------------
# EXTRACTION TEST
# -------------------------
@app.route("/extract_test", methods=["POST"])
def extract_test():
    data = request.json or {}
    filepath = data.get("filepath")
    if not filepath:
        return jsonify({"error": "Provide JSON {\"filepath\": \"/full/path/to/file\"}"}), 400

    try:
        text = extract_text(filepath)
        return jsonify({
            "ok": True,
            "length": len(text),
            "snippet": text[:200]
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# -------------------------
# CHUNKING TEST
# -------------------------
@app.route("/chunk_test", methods=["POST"])
def chunk_test():
    data = request.json or {}
    filepath = data.get("filepath")
    if not filepath:
        return jsonify({"error": "filepath required"}), 400

    try:
        text = extract_text(filepath)
        chunks = prepare_chunks(text)
        return jsonify({
            "ok": True,
            "chunks": len(chunks),
            "chunk_preview": chunks[0][:300] if chunks else ""
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# -------------------------
# INDEXING ENDPOINT (embeddings + faiss) - optimized with cache reuse & logs
# -------------------------
@app.route("/index_file", methods=["POST"])
def index_file():
    data = request.json or {}
    filepath = data.get("filepath")
    prefix = data.get("prefix")
    if not filepath:
        return jsonify({"error": "filepath required"}), 400

    try:
        start_total = time.perf_counter()
        text = extract_text(filepath)
        chunks = prepare_chunks(text)
        if not chunks:
            return jsonify({"error": "no chunks generated"}), 500

        if not prefix:
            prefix = Path(filepath).stem.replace(" ", "_")

        out_dir = OUTPUT_FOLDER
        faiss_path_candidate = out_dir / f"{prefix}_faiss.idx"
        chunks_path_candidate = out_dir / f"{prefix}_chunks.pkl"
        emb_path_candidate = out_dir / f"{prefix}_embeddings.npy"

        # If a FAISS index and chunks already exist, return early (no-op)
        if faiss_path_candidate.exists() and chunks_path_candidate.exists():
            print(f"[index_file] Existing index found for prefix='{prefix}', skipping re-index.")
            elapsed_total = time.perf_counter() - start_total
            return jsonify({
                "ok": True,
                "chunks": len(chunks),
                "faiss_path": str(faiss_path_candidate),
                "chunks_path": str(chunks_path_candidate),
                "embeddings_path": str(emb_path_candidate) if emb_path_candidate.exists() else None,
                "prefix": prefix,
                "info": "index_exists",
                "time_s": elapsed_total
            })

        # Ensure emb_time is defined even if we reuse cache
        emb_time = 0.0

        # Try to reuse cached embeddings if present and length matches
        embeddings = load_embeddings_if_exists(prefix)
        if embeddings is not None and embeddings.shape[0] == len(chunks):
            print(f"[index_file] Reusing cached embeddings for prefix='{prefix}' (vectors={embeddings.shape[0]}).")
        else:
            print(f"[index_file] Computing embeddings for {len(chunks)} chunks...")
            t0 = time.perf_counter()
            embeddings = embed_chunks(chunks, batch_size=64, show_progress=True)
            emb_time = time.perf_counter() - t0
            print(f"[index_file] Embeddings computed in {emb_time:.2f}s, shape={embeddings.shape}.")

        # Build FAISS index (fast)
        t0 = time.perf_counter()
        index = build_faiss_index(embeddings)
        faiss_path, chunks_path, embeddings_path = persist_index(index, chunks, prefix, embeddings=embeddings)
        idx_time = time.perf_counter() - t0

        total_time = time.perf_counter() - start_total
        print(f"[index_file] Indexing complete: chunks={len(chunks)}, emb_time={emb_time:.2f}s, index_time={idx_time:.2f}s, total={total_time:.2f}s")

        return jsonify({
            "ok": True,
            "chunks": len(chunks),
            "faiss_path": faiss_path,
            "chunks_path": chunks_path,
            "embeddings_path": embeddings_path,
            "prefix": prefix,
            "time_s": total_time
        })
    except Exception as e:
        print("[index_file] error:", str(e))
        return jsonify({"ok": False, "error": str(e)}), 500


# -------------------------
# SUMMARIZE endpoint (two-stage tuned, with logging)
# -------------------------
@app.route("/summarize", methods=["POST"])
def summarize_endpoint():
    data = request.json or {}
    prefix = data.get("prefix")
    if not prefix:
        return jsonify({"ok": False, "error": "prefix required (the index prefix)"}), 400

    query = data.get("query", "Produce headlines and a short paragraph under each headline summarising the document.")
    model_name = data.get("model", "gemma3:1b")
    top_k = int(data.get("top_k", 6))
    save_flag = bool(data.get("save", True))

    try:
        start_total = time.perf_counter()
        index, chunks = load_index_and_chunks(prefix)
        n_chunks = len(chunks)
        print(f"[summarize] Starting summarization: prefix={prefix}, n_chunks={n_chunks}, model={model_name}")

        TWO_STAGE_THRESHOLD = 14
        if n_chunks > TWO_STAGE_THRESHOLD:
            # Stage 1: mini summaries in batches
            BATCH_SIZE = 8
            mini_summaries = []
            mini_start = time.perf_counter()
            total_batches = (n_chunks + BATCH_SIZE - 1) // BATCH_SIZE
            for batch_i, i in enumerate(range(0, n_chunks, BATCH_SIZE), start=1):
                batch_chunks = chunks[i:i + BATCH_SIZE]
                batch_context = "\n\n---\n\n".join(batch_chunks)
                batch_prompt = f"""You are an expert summariser. For the context below, produce a single high-quality mini-summary (80-120 words) using only the provided text.

Context:
{batch_context}

Mini-summary (80-120 words):
"""
                t0 = time.perf_counter()
                # call via python client first
                batch_text = None
                try:
                    resp = ollama.run(model_name, batch_prompt)
                    if isinstance(resp, dict):
                        candidate = resp.get("response") or resp.get("output") or ""
                    else:
                        candidate = resp
                    if isinstance(candidate, bytes):
                        candidate = candidate.decode("utf-8", errors="replace")
                    batch_text = str(candidate).strip()
                except Exception:
                    proc = subprocess.run(
                        ["ollama", "run", model_name, "--nowordwrap"],
                        input=batch_prompt.encode("utf-8"),
                        capture_output=True,
                        timeout=300
                    )
                    if proc.returncode != 0:
                        raise RuntimeError(f"ollama CLI failed: {proc.stderr.decode('utf-8', errors='ignore')}")
                    out = proc.stdout
                    if isinstance(out, bytes):
                        out = out.decode("utf-8", errors="replace")
                    batch_text = str(out).strip()

                elapsed = time.perf_counter() - t0
                print(f"[summarize] mini-batch {batch_i}/{total_batches} done in {elapsed:.2f}s")
                if isinstance(batch_text, bytes):
                    batch_text = batch_text.decode("utf-8", errors="replace")
                mini_summaries.append(batch_text)
                time.sleep(0.08)

            mini_time = time.perf_counter() - mini_start
            print(f"[summarize] All mini-summaries done ({len(mini_summaries)}) in {mini_time:.2f}s")

            # Stage 2: aggregate mini summaries into final structured summary
            combined_text = "\n\n".join([s if isinstance(s, str) else s.decode("utf-8", errors="replace") for s in mini_summaries])
            aggregate_prompt = f"""
You are an expert editor. Using ONLY the combined mini-summaries below, produce a final structured summary.

Requirements:
- 10–15 short, meaningful headlines.
- Under each headline write a substantive paragraph (100–160 words).
- Finish with a "Conclusion" section (350–500 words).
- Base content only on the provided mini-summaries.
Combined mini-summaries:
{combined_text}

Final structured summary:
"""
            t0 = time.perf_counter()
            try:
                resp2 = ollama.run(model_name, aggregate_prompt)
                if isinstance(resp2, dict):
                    candidate = resp2.get("response") or resp2.get("output") or ""
                else:
                    candidate = resp2
                if isinstance(candidate, bytes):
                    candidate = candidate.decode("utf-8", errors="replace")
                final_text = str(candidate).strip()
            except Exception:
                proc = subprocess.run(
                    ["ollama", "run", model_name, "--nowordwrap"],
                    input=aggregate_prompt.encode("utf-8"),
                    capture_output=True,
                    timeout=600
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"ollama CLI failed: {proc.stderr.decode('utf-8', errors='ignore')}")
                out = proc.stdout
                if isinstance(out, bytes):
                    out = out.decode("utf-8", errors="replace")
                final_text = str(out).strip()
            agg_time = time.perf_counter() - t0
            print(f"[summarize] Aggregate done in {agg_time:.2f}s")
        else:
            # small doc: single-shot summarization using top_k retrieval
            selected_chunks, indices = retrieve_top_k(prefix, query, top_k=top_k)
            context = assemble_context(selected_chunks, max_chars=6500)
            direct_prompt = f"""You are an expert summariser. Given the context below produce structured headlines and for each headline a substantive paragraph (90-150 words). End with a 'Conclusion' (150–250 words). Use only the context.

Context:
{context}
"""
            t0 = time.perf_counter()
            try:
                resp = ollama.run(model_name, direct_prompt)
                if isinstance(resp, dict):
                    candidate = resp.get("response") or resp.get("output") or ""
                else:
                    candidate = resp
                if isinstance(candidate, bytes):
                    candidate = candidate.decode("utf-8", errors="replace")
                final_text = str(candidate).strip()
            except Exception:
                proc = subprocess.run(
                    ["ollama", "run", model_name, "--nowordwrap"],
                    input=direct_prompt.encode("utf-8"),
                    capture_output=True,
                    timeout=600
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"ollama CLI failed: {proc.stderr.decode('utf-8', errors='ignore')}")
                out = proc.stdout
                if isinstance(out, bytes):
                    out = out.decode("utf-8", errors="replace")
                final_text = str(out).strip()
            print(f"[summarize] Direct summarization done in {time.perf_counter() - t0:.2f}s")

        # save final
        summary_filename = f"{prefix}_summary.md"
        if save_flag:
            out_path = OUTPUT_FOLDER / summary_filename
            out_path.write_text(final_text, encoding="utf-8")
            download_url = f"/download_summary/{summary_filename}"
        else:
            download_url = None

        total_elapsed = time.perf_counter() - start_total
        print(f"[summarize] Completed summarization for prefix='{prefix}' total_time={total_elapsed:.2f}s")

        return jsonify({
            "ok": True,
            "summary": final_text,
            "download_url": download_url,
            "prefix": prefix,
            "chunks": n_chunks,
            "time_s": total_elapsed
        })
    except Exception as e:
        print("[summarize] error:", str(e))
        return jsonify({"ok": False, "error": str(e)}), 500


# -------------------------
# ASK endpoint (RAG Q&A)
# -------------------------
def extract_pages_from_chunk_text(chunk_text: str) -> List[int]:
    """Return a list of page numbers found in the chunk text (if any)."""
    pages = [int(m) for m in re.findall(r"\[page:(\d+)\]", chunk_text)]
    if pages:
        return sorted(set(pages))
    slides = [int(m) for m in re.findall(r"\[slide:(\d+)\]", chunk_text)]
    if slides:
        return sorted(set(slides))
    return []


@app.route("/ask", methods=["POST"])
def ask_endpoint():
    """
    POST JSON:
    {
      "prefix": "sameer_resume",   # required: index prefix
      "question": "What is virtualization?",
      "model": "gemma3:1b",        # optional
      "top_k": 5                  # optional
    }
    """
    data = request.json or {}
    prefix = data.get("prefix")
    question = data.get("question")
    if not prefix or not question:
        return jsonify({"ok": False, "error": "prefix and question required"}), 400

    model_name = data.get("model", "gemma3:1b")
    top_k = int(data.get("top_k", 5))

    try:
        # retrieve top_k chunks (uses SentenceTransformer + FAISS)
        selected_chunks, indices = retrieve_top_k(prefix, question, top_k=top_k)
        if not selected_chunks:
            return jsonify({"ok": False, "error": "no chunks retrieved for this prefix"}), 500

        # build context including short markers for page numbers if present
        context_blocks = []
        for idx, chunk in zip(indices, selected_chunks):
            pages = extract_pages_from_chunk_text(chunk)
            page_hint = f"(pages: {pages[0]}-{pages[-1]})" if pages and len(pages) > 1 else (f"(page: {pages[0]})" if pages else "")
            # include index and page hint to help model cite
            context_blocks.append(f"CHUNK_INDEX:{idx} {page_hint}\n{chunk}")

        context = "\n\n---\n\n".join(context_blocks)

        prompt = f"""You are a precise assistant answering questions using ONLY the provided context. Do NOT hallucinate or add information not present in the context. If the document does not contain the answer, respond: "The document does not contain the requested information."

Context:
{context}

Question:
{question}

Answer concisely. At the end, provide a short "SOURCES" line listing page numbers or chunk indices used, e.g. "SOURCES: pages 12-14" or "SOURCES: CHUNK_INDEX:3".
"""
        # call model
        try:
            resp = ollama.run(model_name, prompt)
            if isinstance(resp, dict):
                candidate = resp.get("response") or resp.get("output") or ""
            else:
                candidate = resp
            if isinstance(candidate, bytes):
                candidate = candidate.decode("utf-8", errors="replace")
            answer_text = str(candidate)
        except Exception:
            with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
                tf.write(prompt); tf.flush()
                proc = subprocess.run(["ollama", "run", model_name, "--nowordwrap"], stdin=open(tf.name, "rb"), capture_output=True, text=True, timeout=300)
                if proc.returncode != 0:
                    return jsonify({"ok": False, "error": "ollama CLI failed", "stderr": proc.stderr}), 500
                answer_text = proc.stdout.strip()

        # Attempt to extract SOURCES line from model output
        sources = []
        m = re.search(r"SOURCES\:\s*(.*)$", answer_text, re.I | re.M)
        if m:
            sources_text = m.group(1).strip()
            sources = [s.strip() for s in re.split(r"[,;]", sources_text) if s.strip()]
        else:
            # fallback: use the pages extracted from selected chunks
            pages_set = set()
            for ch in selected_chunks:
                pages = extract_pages_from_chunk_text(ch)
                for p in pages:
                    pages_set.add(p)
            if pages_set:
                pages_list = sorted(pages_set)
                # produce a compact pages range or list
                if len(pages_list) > 1:
                    sources = [f"pages {pages_list[0]}-{pages_list[-1]}"]
                else:
                    sources = [f"page {pages_list[0]}"]

        return jsonify({
            "ok": True,
            "answer": answer_text,
            "sources": sources,
            "used_chunk_indices": indices,
            "prefix": prefix
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# -------------------------
# DOWNLOAD summary file
# -------------------------
@app.route("/download_summary/<filename>", methods=["GET"])
def download_summary(filename):
    safe = Path(filename).name
    return send_from_directory(str(OUTPUT_FOLDER), safe, as_attachment=True)


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)