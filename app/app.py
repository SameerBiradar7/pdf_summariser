import os
import time
import subprocess
import tempfile
from pathlib import Path

from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# extraction / chunking / embeddings / retrieval helpers
from app.extract import extract_text
from app.chunking import prepare_chunks
from app.embeddings import embed_chunks, build_faiss_index, persist_index, load_index_and_chunks
from app.retrieval import retrieve_top_k, assemble_context

import ollama

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(BASE_DIR / "app" / "templates"))
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

ALLOWED_EXT = {".pdf", ".txt", ".pptx", ".ppt", ".docx", ".md"}


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
# Summarize endpoint (two-stage for large docs) + persistence (with Conclusion)
# -------------------------
@app.route("/summarize", methods=["POST"])
def summarize_endpoint():
    """
    POST JSON:
      {
        "prefix": "sameer_resume",    # required
        "query": "Summarize ...",     # optional
        "model": "gemma3:1b",         # optional
        "top_k": 5,                   # optional
        "save": true                  # optional, save summary to outputs/<prefix>_summary.txt
      }
    """
    data = request.json or {}
    prefix = data.get("prefix")
    if not prefix:
        return jsonify({"ok": False, "error": "prefix required (the index prefix)"}), 400

    query = data.get("query", "Produce headlines and a short paragraph under each headline summarising the document.")
    model_name = data.get("model", "gemma3:1b")
    top_k = int(data.get("top_k", 5))
    save_flag = bool(data.get("save", True))

    try:
        # load entire index and chunks (we will use chunks for two-stage)
        index, chunks = load_index_and_chunks(prefix)
        n_chunks = len(chunks)

        # If document is large, do two-stage summarisation:
        # Stage A: produce short summaries for chunk groups
        # Stage B: aggregate summaries and ask model to produce headlines + substantial content + Conclusion
        TWO_STAGE_THRESHOLD = 12   # tuneable
        grouped_summaries = []

        if n_chunks > TWO_STAGE_THRESHOLD:
            # group chunks into manageable batches to summarise
            BATCH_SIZE = 6
            for i in range(0, n_chunks, BATCH_SIZE):
                batch_chunks = chunks[i:i+BATCH_SIZE]
                # assemble batch context
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
                # call model
                try:
                    resp = ollama.run(model_name, batch_prompt)
                    if isinstance(resp, dict):
                        text_out = resp.get("response") or resp.get("output") or str(resp)
                    else:
                        text_out = str(resp)
                except Exception:
                    # fallback to CLI
                    with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
                        tf.write(batch_prompt); tf.flush()
                        proc = subprocess.run(["ollama", "run", model_name, "--nowordwrap"], stdin=open(tf.name,"rb"), capture_output=True, text=True, timeout=300)
                        if proc.returncode != 0:
                            raise RuntimeError(f"ollama CLI failed: {proc.stderr}")
                        text_out = proc.stdout.strip()

                grouped_summaries.append(text_out)
                # brief pause to avoid hammering local model
                time.sleep(0.2)

            # Combine grouped summaries
            combined = "\n\n".join(grouped_summaries)
            # Stage B: aggregate grouped summaries into headlines + more substantive content and a long Conclusion
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
            # small document: retrieve top_k chunks and summarise directly in one pass with richer paragraphs + Conclusion
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

        # Optionally save final summary file
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
# Download endpoint for saved summary files
# -------------------------
@app.route("/download_summary/<filename>", methods=["GET"])
def download_summary(filename):
    # Security: restrict to outputs folder
    safe = Path(filename).name
    return send_from_directory(str(OUTPUT_FOLDER), safe, as_attachment=True)


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)