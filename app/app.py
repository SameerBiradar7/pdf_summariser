# app/app.py
import os
import time
import subprocess
import tempfile
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import ollama

# local modules
from app.extract import extract_text
from app.chunking import prepare_chunks
from app.embeddings import embed_chunks, build_faiss_index, persist_index, load_index_and_chunks
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


# -------------------------
# HOME
# -------------------------
@app.route("/", methods=["GET"])
def index():
    models = ["gemma3:1b", "qwen2:1.5b", "llama3.2:latest"]
    return render_template("index.html", models=models, default_model=models[0])


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
# INDEXING ENDPOINT (embeddings + faiss)
# -------------------------
@app.route("/index_file", methods=["POST"])
def index_file():
    data = request.json or {}
    filepath = data.get("filepath")
    prefix = data.get("prefix")
    if not filepath:
        return jsonify({"error": "filepath required"}), 400

    try:
        text = extract_text(filepath)
        chunks = prepare_chunks(text)
        if not chunks:
            return jsonify({"error": "no chunks generated"}), 500

        embs = embed_chunks(chunks)
        index = build_faiss_index(embs)
        if not prefix:
            prefix = Path(filepath).stem.replace(" ", "_")
        faiss_path, chunks_path = persist_index(index, chunks, prefix)

        return jsonify({
            "ok": True,
            "chunks": len(chunks),
            "faiss_path": faiss_path,
            "chunks_path": chunks_path,
            "prefix": prefix
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# -------------------------
# SUMMARIZE endpoint (existing logic)
# -------------------------
@app.route("/summarize", methods=["POST"])
def summarize_endpoint():
    data = request.json or {}
    prefix = data.get("prefix")
    if not prefix:
        return jsonify({"ok": False, "error": "prefix required (the index prefix)"}), 400

    query = data.get("query", "Produce headlines and a short paragraph under each headline summarising the document.")
    model_name = data.get("model", "gemma3:1b")
    top_k = int(data.get("top_k", 5))
    save_flag = bool(data.get("save", True))

    try:
        index, chunks = load_index_and_chunks(prefix)
        n_chunks = len(chunks)

        TWO_STAGE_THRESHOLD = 12
        grouped_summaries = []

        if n_chunks > TWO_STAGE_THRESHOLD:
            BATCH_SIZE = 6
            for i in range(0, n_chunks, BATCH_SIZE):
                batch_chunks = chunks[i:i+BATCH_SIZE]
                batch_context = "\n\n---\n\n".join(batch_chunks)
                batch_prompt = f"""You are a concise summarizer. For the context below produce a short single-sentence summary for each section.
Context:
{batch_context}

Instructions:
- For each section in the context, produce a one-sentence summary.
- Prefix each summary with "SECTION_{i}:" where i is the sequential index within this batch (or a label you choose).
Output format:
SECTION_<index>: <one-sentence summary>
"""
                try:
                    resp = ollama.run(model_name, batch_prompt)
                    if isinstance(resp, dict):
                        text_out = resp.get("response") or resp.get("output") or str(resp)
                    else:
                        text_out = str(resp)
                except Exception:
                    with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
                        tf.write(batch_prompt); tf.flush()
                        proc = subprocess.run(["ollama", "run", model_name, "--nowordwrap"], stdin=open(tf.name,"rb"), capture_output=True, text=True, timeout=300)
                        if proc.returncode != 0:
                            raise RuntimeError(f"ollama CLI failed: {proc.stderr}")
                        text_out = proc.stdout.strip()

                grouped_summaries.append(text_out)
                time.sleep(0.2)

            combined = "\n\n".join(grouped_summaries)
            aggregate_prompt = f"""
You are an expert editor. Given the following many one-line summaries (compact), produce a structured document with:
- 8-12 clear headlines (short, descriptive)
- Under each headline give a substantive paragraph (approx. 70-140 words) that summarizes the relevant content
- After all the headlines, add a final section titled "Conclusion" with a 120–200 word overview of the entire document

Be concise and factual. Use the content below as the only source.

Content:
{combined}

Output format:
Headline 1:
<paragraph (70-140 words)>

Headline 2:
<paragraph (70-140 words)>

...

Conclusion:
<120–200 word overall summary of the full PDF/PPT>
"""
            try:
                resp2 = ollama.run(model_name, aggregate_prompt)
                if isinstance(resp2, dict):
                    final_text = resp2.get("response") or resp2.get("output") or str(resp2)
                else:
                    final_text = str(resp2)
            except Exception:
                with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
                    tf.write(aggregate_prompt); tf.flush()
                    proc = subprocess.run(["ollama", "run", model_name, "--nowordwrap"], stdin=open(tf.name,"rb"), capture_output=True, text=True, timeout=600)
                    if proc.returncode != 0:
                        raise RuntimeError(f"ollama CLI failed: {proc.stderr}")
                    final_text = proc.stdout.strip()
        else:
            selected_chunks, indices = retrieve_top_k(prefix, query, top_k=top_k)
            context = assemble_context(selected_chunks, max_chars=6500)
            direct_prompt = f"""
You are an expert summariser. Given the context below produce structured headlines and for each headline a substantive paragraph (70-140 words). At the end, add a "Conclusion" section (120-200 words) that summarizes the whole document. Use only the context.
Context:
{context}

Instructions:
- Provide 6-10 headlines with substantive paragraphs.
- Be factual and concise.
- End with a "Conclusion" section (120-200 words).
"""
            try:
                resp = ollama.run(model_name, direct_prompt)
                if isinstance(resp, dict):
                    final_text = resp.get("response") or resp.get("output") or str(resp)
                else:
                    final_text = str(resp)
            except Exception:
                with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
                    tf.write(direct_prompt); tf.flush()
                    proc = subprocess.run(["ollama", "run", model_name, "--nowordwrap"], stdin=open(tf.name,"rb"), capture_output=True, text=True, timeout=600)
                    if proc.returncode != 0:
                        raise RuntimeError(f"ollama CLI failed: {proc.stderr}")
                    final_text = proc.stdout.strip()

        summary_filename = f"{prefix}_summary.txt"
        if save_flag:
            out_path = OUTPUT_FOLDER / summary_filename
            out_path.write_text(final_text, encoding="utf-8")
            download_url = f"/download_summary/{summary_filename}"
        else:
            download_url = None

        return jsonify({
            "ok": True,
            "summary": final_text,
            "download_url": download_url,
            "prefix": prefix,
            "chunks": n_chunks
        })
    except Exception as e:
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
        selected_chunks, indices = retrieve_top_k(prefix, [], top_k=top_k) if False else retrieve_top_k(prefix, question, top_k=top_k)
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
            answer_text = resp.get("response") if isinstance(resp, dict) else str(resp)
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