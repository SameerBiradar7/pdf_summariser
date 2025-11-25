# AI Document Summariser (Local RAG + Flask + Ollama + MongoDB Atlas)

A full-stack, AI-powered **document understanding system** that performs:

- Secure file uploads  
- Text extraction from PDF, PPTX, DOCX, TXT  
- Chunking + embeddings  
- Vector indexing using FAISS  
- Local LLM-based summarisation (Ollama)  
- Document-grounded Q&A (RAG)  
- Per-user summary storage in **MongoDB Atlas**  
- Beautiful animated UI  
- (Future) Cloud storage on AWS S3 with one-time download links  

This project runs fully locally for AI inference while storing user metadata and summaries persistently in the cloud.
## 🚀 Core Features

### ✔️ File Upload System
Supports all major document types. Uses secure form uploads via Flask.

### ✔️ Text Extraction
Extraction pipeline automatically detects the file type and uses:
- `pdfplumber` for PDFs  
- `python-pptx` for PPTX  
- `python-docx` for DOCX  
- Plain-text fallback  

Includes **page number tagging** (`[page: X]`) for accurate RAG retrieval.

### ✔️ Chunking + Embeddings
Documents are chunked into optimal sizes (400–1000 chars).  
Embeddings are generated using **SentenceTransformer (MiniLM)**.

### ✔️ FAISS Vector Store
All embeddings are stored in a local FAISS index:
Enables fast retrieval for summarization and Q&A.

### ✔️ Local LLMs via Ollama
Your system supports **multiple local models**:
- `gemma3:1b`  
- `qwen2:1.5b`  
- `llama3.2:latest`

Advantages:
- No API cost  
- No internet dependency  
- High privacy  

### ✔️ Accurate RAG Summarisation
A two-stage pipeline:

1. Retrieve relevant chunks  
2. Local LLM generates:
   - Headings  
   - Condensed explanations  
   - A final structured conclusion  

### ✔️ Document Q&A
Users can query ANY part of the uploaded document.

The system:
1. Retrieves relevant chunks  
2. Passes them to local LLM  
3. Generates grounded answers  
4. Returns **page citations**  

### ✔️ Modern Animated UI
- CSS glassmorphism  
- Smooth transitions  
- Responsive layout  
- Separate Upload Panel & Ask-Question Panel  
- Clear intermediate statuses: Upload → Index → Summarize  

Screenshot reference:  
`file:///mnt/data/Screenshot 2025-11-25 at 12.35.07.png`

# 🗄️ MongoDB Atlas (Core Component)

MongoDB Atlas is used as the **persistent database layer** of the project.

### What is stored in MongoDB Atlas?

#### 1. User Accounts  
- email  
- password (argon2 hash)  
- verification flag  
- created_at  
- last_login_at  

#### 2. User Uploads  
Metadata such as:
- filename  
- file prefix  
- file path  
- upload timestamp  

#### 3. User Summaries  
Every summary generated is also stored:

- summary text  
- model used  
- chunk count  
- pages referenced  
- timestamp  
- links for download  
- parent upload reference  

This enables users to log in from anywhere and access/download their previous summaries.

MongoDB Atlas is now an **active project component**, not a future enhancement.

# 🧱 Architecture Overview

Frontend  →  Flask Backend  →  RAG Pipeline  →  Local LLM (Ollama)
↓
FAISS Vector DB
↓
MongoDB Atlas (User + Metadata)

## 📁 Project Structure
pdf_summariser/
│
├── app/
│   ├── app.py
│   ├── extract.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── templates/index.html
│
├── uploads/       # user files (dev only)
├── outputs/       # FAISS index + summaries (dev only)
├── run.sh
└── README.md

For production: uploads and outputs will move to S3 (see future enhancements).

# 🔐 Authentication (Current Design)

Authentication is designed using:

- **MongoDB Atlas** for storage  
- **Argon2 hashing** for passwords  
- **Flask-Login** for session handling  
- **Email verification** (token-based)  
- **Session-based security** for UI users  

This enables multiple users to have their own workspace:
- Personal file uploads  
- Personal summaries  
- Personal Q&A history  

# ☁️ Future Enhancements (AWS S3 Only)

### 🚀 1. Move File Storage to AWS S3
Instead of saving files locally, upload documents and summaries to S3.

Planned S3 structure:
s3://bucket/uploads/<user_id>//
s3://bucket/summaries/<user_id>//summary.txt

### 🔐 2. One-Time Download Links (Signed URLs)
- A summary or document can be downloaded **only once**  
- URL expires automatically after N minutes  
- Great for security and bandwidth control  

### 🧽 3. Auto-Deletion Policies
Using S3 lifecycle rules:
- Delete unused summary files after X days  
- Clean orphaned files automatically  

These enhancements convert the system into a **full AI cloud-integrated summariser**.

# 🏁 Summary

This project is a complete **AI-powered document intelligence system** with:

- Modern animated frontend  
- Flask backend  
- Local LLM summarisation  
- RAG-based Q&A  
- MongoDB Atlas user accounts + summary storage  
- FAISS vector indexing  
- Expandable architecture  

And with S3 integration coming next, it becomes a **cloud-backed AI summariser platform** for
everyday or enterprise use.
