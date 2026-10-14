# ==============================================================================
# YouTube Lecture Assistant - Phase 2: Basic In-Memory RAG Pipeline
# ==============================================================================
# In Phase 2, we introduce the full RAG (Retrieval-Augmented Generation) flow:
#   1. Transcript Extraction  (Phase 1 preserved)
#   2. Chunking               (Break long text into smaller pieces)
#   3. Retrieval              (Find chunks matching keywords from user question)
#   4. Augmentation           (Combine matched chunks + question into a prompt)
#   5. Generation             (Ask Groq / Llama 3.3 to answer using only the context)
# ==============================================================================

import os
import re

# Load .env file automatically so we never expose API keys in code
from dotenv import load_dotenv

# Third-party library for retrieving YouTube transcripts (from Phase 1)
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

# Groq SDK for calling LLM (free, ultra-fast, no credit card needed)
from groq import Groq

# Read variables from .env into Python's environment
load_dotenv()


# ------------------------------------------------------------------------------
# PHASE 1 FUNCTIONS (PRESERVED)
# ------------------------------------------------------------------------------

def extract_video_id(url: str) -> str:
    """
    Extracts the unique 11-character Video ID from various YouTube URL formats.
    """
    pattern = r'(?:v=|\/embed\/|\/shorts\/|youtu\.be\/|^)([a-zA-Z0-9_-]{11})(?:[&?/\s]|$)'
    match = re.search(pattern, url.strip())
    if match:
        return match.group(1)
    return None


def get_youtube_transcript(video_id: str) -> str:
    """
    Fetches transcript chunks from YouTube and combines them into one readable text.
    """
    try:
        api = YouTubeTranscriptApi()
        transcript_data = api.fetch(video_id)
        full_transcript = " ".join(item.text.replace("\n", " ") for item in transcript_data)
        return full_transcript
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        raise
    except Exception as e:
        raise RuntimeError(f"Unexpected error while fetching transcript: {e}")


# ------------------------------------------------------------------------------
# PHASE 2 FUNCTIONS: CHUNKING, RETRIEVAL, AND GENERATION
# ------------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list:
    """
    Splits continuous text into overlapping chunks of words.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break

    return chunks


def retrieve_relevant_chunks(chunks: list, question: str, top_k: int = 3) -> list:
    """
    Retrieval step: Finds the chunks in memory that are most relevant to the question.
    """
    stop_words = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or",
        "is", "are", "was", "were", "what", "how", "why", "when", "where",
        "who", "which", "do", "does", "did", "can", "could", "should", "would",
        "it", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
    }

    raw_keywords = re.findall(r"\b\w+\b", question.lower())
    keywords = [kw for kw in raw_keywords if kw not in stop_words]

    if not keywords:
        keywords = raw_keywords

    scored_chunks = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = 0
        for kw in keywords:
            score += len(re.findall(r"\b" + re.escape(kw) + r"\b", chunk_lower))
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    matching_chunks = [chunk for score, chunk in scored_chunks[:top_k] if score > 0]

    if not matching_chunks and chunks:
        return [chunks[0]]

    return matching_chunks


def generate_answer(context_chunks: list, question: str, api_key: str = None) -> str:
    """
    Generation step: Sends the retrieved chunks and the question to Groq (Llama 3.3).
    Instructs the LLM to strictly base its answer on the provided lecture context.
    """
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("Missing GROQ_API_KEY. Please set your key in .env")

    context_text = "\n\n--- NEXT CHUNK ---\n\n".join(context_chunks)

    prompt = f"""You are a helpful YouTube Lecture Assistant.
Answer the user's question using ONLY the provided lecture transcript context below.

STRICT RULES:
1. If the answer is found in the provided context, give a clear, accurate, and concise answer.
2. If the answer CANNOT be found in the provided context, answer with:
   "I cannot answer this question based on the provided lecture transcript."
3. Do not make up facts or use external knowledge outside the provided lecture context.

--- LECTURE CONTEXT ---
{context_text}

--- USER QUESTION ---
{question}
"""

    client = Groq(api_key=api_key)
    
    # Try high performance models available on your Groq account
    models_to_try = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
    for model_name in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            return response.choices[0].message.content
        except Exception:
            continue

    # Final fallback if loop ends
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ------------------------------------------------------------------------------
# MAIN PIPELINE CONTROLLER
# ------------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("   YouTube Lecture Assistant - Phase 2: In-Memory RAG System")
    print("=" * 65)

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or api_key == "your_groq_api_key_here":
        print("\n[Configuration Notice] Missing GROQ_API_KEY in .env file.")
        print("To enable the AI to answer questions, you need a free Groq API key:")
        print("  1. Get a free key at: https://console.groq.com/keys")
        print("  2. Open the '.env' file in this project folder.")
        print("  3. Set: GROQ_API_KEY=your_actual_key_here")
        print("\nPlease add your API key to .env and run this program again!\n")
        return

    # Step 1: Prompt user for a YouTube URL
    user_input = input("\nEnter YouTube video URL (or Video ID): ").strip()
    if not user_input:
        print("\n[Error] No input provided. Please run again with a valid link.")
        return

    # Step 2: Extract video ID
    video_id = extract_video_id(user_input)
    if not video_id:
        print(f"\n[Error] Could not find a valid YouTube Video ID in: '{user_input}'")
        return

    print(f"\n[1/4] Found Video ID: {video_id}")
    print("[2/4] Downloading transcript from YouTube...")

    # Step 3: Fetch transcript
    try:
        raw_transcript = get_youtube_transcript(video_id)
        word_count = len(raw_transcript.split())
        print(f"[2/4] Success! Retrieved {word_count:,} words from the lecture.")
    except TranscriptsDisabled:
        print("\n[Error] Transcripts are turned off for this video. Please try another video.")
        return
    except NoTranscriptFound:
        print("\n[Error] No transcripts/captions available for this video.")
        return
    except VideoUnavailable:
        print("\n[Error] Video is unavailable (private, deleted, or invalid link).")
        return
    except Exception as error:
        print(f"\n[Error] Could not retrieve transcript: {error}")
        return

    # Step 4: Split into chunks in memory
    print("[3/4] Chunking lecture transcript into bite-sized pieces in memory...")
    chunks = chunk_text(raw_transcript, chunk_size=150, overlap=30)
    print(f"[3/4] Created {len(chunks)} chunks in memory (~150 words each).")

    print("[4/4] RAG pipeline ready!")
    print("\nYou can now ask any question about this lecture!")
    print("Type 'exit' or 'quit' whenever you want to finish.\n")

    # Interactive Q&A loop
    while True:
        try:
            print("-" * 65)
            question = input("\nYour Question: ").strip()

            if not question:
                continue

            if question.lower() in ("exit", "quit", "q"):
                print("\nThank you for using YouTube Lecture Assistant!")
                break

            print("\n[RAG Step 1: Retrieval] Searching chunks for relevant keywords...")
            relevant_chunks = retrieve_relevant_chunks(chunks, question, top_k=3)
            print(f"[RAG Step 1: Retrieval] Found {len(relevant_chunks)} relevant chunk(s).")

            print("[RAG Step 2: Augmentation & Generation] Asking Groq (Llama 3.3)...")
            answer = generate_answer(relevant_chunks, question, api_key)

            print("\n" + "=" * 65)
            print("AI ANSWER:")
            print("=" * 65)
            print(answer.strip())
            print("=" * 65)

        except Exception as error:
            print(f"\n[Error during Q&A] {error}")


if __name__ == "__main__":
    main()
