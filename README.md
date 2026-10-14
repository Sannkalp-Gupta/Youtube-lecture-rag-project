# 🎓 YouTube Lecture Assistant (RAG Pipeline)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20Llama%203.3-f55036.svg)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An in-memory **Retrieval-Augmented Generation (RAG)** pipeline that extracts transcripts from educational YouTube videos, chunks them with sliding-window overlap, performs keyword retrieval, and answers user questions with strictly grounded responses powered by **Groq** high-speed inference.

---

## 🏗️ Architecture & Pipeline Flow

```
[ YouTube URL ]
       │
       ▼
[ 1. Video ID Extraction & Transcript Fetcher ]
  └─ Extract 11-char ID -> Fetch caption track via youtube-transcript-api
       │
       ▼
[ 2. Chunking Engine (Sliding Window) ]
  └─ 150-word chunks with 30-word overlap (preserves contextual continuity)
       │
       ▼
[ 3. In-Memory Keyword Retrieval ]
  └─ Stopword filtration -> Frequency scoring -> Top-K relevant chunks
       │
       ▼
[ 4. Grounded Prompt Augmentation ]
  └─ Context injection + Hallucination prevention rules
       │
       ▼
[ 5. Generation (Groq LLM) ]
  └─ Sub-second inference via Groq Cloud
       │
       ▼
[ 6. Presentation Layer ]
  ├─ Streamlit Web UI (Interactive browser app)
  └─ Terminal CLI (Lightweight developer shell)
```

---

## 🌟 Key Features & Engineering Highlights

- **Strict Grounding (Anti-Hallucination Guardrails)**: The prompt instructs the model to answer *only* using information present in the transcript context, explicitly stating when a topic is not covered in the lecture.
- **Sliding-Window Chunking**: Splits transcripts into 150-word segments with 30-word overlap, preventing vital context from being split across chunk boundaries.
- **Ultra-Fast Inference**: Integrated with Groq's LPU-accelerated cloud for sub-second responses.
- **Dual Interface**:
  - **Web UI**: Interactive dashboard built with Streamlit.
  - **CLI**: Lightweight terminal script for headless environments.
- **Production-Ready Secrets Management**: Secrets are isolated in `.env` and kept strictly out of version control via `.gitignore`.

---

## 📁 Repository Structure

```text
├── app.py              # Streamlit Web UI application
├── main.py             # Core RAG pipeline logic + CLI runner
├── requirements.txt    # Lean list of production dependencies
├── .env.example        # Template for required environment variables
├── .gitignore          # Excludes venv, .env secrets, and cache artifacts
└── README.md           # Project documentation and architecture guide
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10 or higher
- A free Groq API key from [console.groq.com/keys](https://console.groq.com/keys)

### 2. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/youtube-lecture-rag.git
cd youtube-lecture-rag
```

### 3. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate    # On macOS/Linux
# venv\Scripts\activate     # On Windows
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure API Key
Duplicate the example environment file and add your Groq key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GROQ_API_KEY=gsk_your_actual_key_here
```

---

## 🖥️ Running the Application

### Option A: Streamlit Web Interface (Recommended)
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

1. Paste any YouTube video URL with available captions.
2. Click **Process Video**.
3. Type your question and view the grounded AI answer.

### Option B: Terminal CLI
```bash
python3 main.py
```
Run an interactive Q&A loop directly in your command line.

---

## 🛠️ Tech Stack

- **Inference Engine**: [Groq Cloud](https://groq.com/)
- **Transcript Extraction**: [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api)
- **Web Interface**: [Streamlit](https://streamlit.io/)
- **Configuration**: [python-dotenv](https://github.com/theskumar/python-dotenv)

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
