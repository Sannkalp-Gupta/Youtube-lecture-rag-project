# ==============================================================================
# YouTube Lecture Assistant - Phase 3: Streamlit Web Interface
# ==============================================================================
# In Phase 3, we connect our working RAG pipeline to a clean web browser interface
# using Streamlit and Groq.
#
# The 5 Core UI Components:
#   1. YouTube URL input box
#   2. "Process Video" button
#   3. Question input box
#   4. "Ask Question" button
#   5. Answer display area
# ==============================================================================

import os
import sys
from dotenv import load_dotenv

# Try importing Streamlit (installed on Streamlit Cloud)
try:
    import streamlit as st  # type: ignore
except ImportError:
    print("\n" + "=" * 65)
    print("   Notice: Streamlit is hosted in the Cloud!")
    print("   👉 Open your live Web App at:")
    print("      https://youtube-lecture-rag-project.streamlit.app")
    print("\n   To run the local terminal assistant on your Mac:")
    print("      python3 main.py")
    print("=" * 65 + "\n")
    sys.exit(0)

# Import the working functions from Phase 1 & 2 in main.py
from main import (
    extract_video_id,
    get_youtube_transcript,
    chunk_text,
    retrieve_relevant_chunks,
    generate_answer,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

# Load secret API key from our .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Set the browser window title and favicon icon
st.set_page_config(page_title="YouTube Lecture Assistant", page_icon="🎓")

# Display the main header on the page
st.title("🎓 YouTube Lecture Assistant")
st.write("Ask questions about any YouTube lecture using Groq LLM and RAG.")

# Check if the API key exists in .env
if not api_key or api_key == "your_groq_api_key_here":
    st.error("⚠️ GROQ_API_KEY is missing!")
    st.markdown("""
    Please open your **`.env`** file and paste your Groq API key:
    ```env
    GROQ_API_KEY=gsk_...
    ```
    Get a free key instantly at: [console.groq.com/keys](https://console.groq.com/keys)
    """)
    st.stop()

# Initialize Streamlit's in-memory notepad (session_state)
# This keeps the lecture chunks saved in memory between button clicks
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []

if "video_title" not in st.session_state:
    st.session_state["video_title"] = ""


# ------------------------------------------------------------------------------
# STEP 1 & 2: YOUTUBE URL INPUT AND "PROCESS VIDEO" BUTTON
# ------------------------------------------------------------------------------
st.subheader("1. Load Lecture Video")

# UI Element 1: YouTube URL text input box
youtube_url = st.text_input(
    "Enter YouTube Video URL:",
    placeholder="https://www.youtube.com/watch?v=..."
)

# UI Element 2: "Process Video" button
if st.button("Process Video"):
    if not youtube_url.strip():
        st.warning("Please enter a YouTube URL first.")
    else:
        # Step A: Extract video ID from URL
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("Could not find a valid YouTube video ID. Please check the URL.")
        else:
            try:
                # Step B: Fetch transcript with a loading spinner
                with st.spinner("Downloading transcript from YouTube..."):
                    raw_transcript = get_youtube_transcript(video_id)

                # Step C: Chunk the transcript in memory
                with st.spinner("Splitting transcript into small chunks..."):
                    chunks = chunk_text(raw_transcript, chunk_size=150, overlap=30)
                    # Save the chunks in Streamlit session memory
                    st.session_state["chunks"] = chunks
                    st.session_state["video_title"] = youtube_url

                word_count = len(raw_transcript.split())
                st.success(f"Video processed successfully! ({word_count:,} words divided into {len(chunks)} chunks in memory)")

            except TranscriptsDisabled:
                st.error("Transcripts/subtitles are turned off for this video. Please try another video.")
            except NoTranscriptFound:
                st.error("No transcript or subtitles could be found for this video.")
            except VideoUnavailable:
                st.error("The video is unavailable. It may be private or deleted.")
            except Exception as e:
                st.error(f"Error processing video: {e}")

st.divider()


# ------------------------------------------------------------------------------
# STEP 3, 4 & 5: QUESTION INPUT, "ASK QUESTION" BUTTON, AND ANSWER DISPLAY
# ------------------------------------------------------------------------------
st.subheader("2. Ask a Question")

# UI Element 3: Question text input box
user_question = st.text_input(
    "Ask a question about the lecture:",
    placeholder="e.g., What is the main conclusion of this lecture?"
)

# UI Element 4: "Ask Question" button
if st.button("Ask Question"):
    # Guard check: Ensure video has been processed first
    if not st.session_state["chunks"]:
        st.warning("Please process a YouTube video above before asking questions.")
    elif not user_question.strip():
        st.warning("Please type a question before clicking Ask.")
    else:
        try:
            with st.spinner("Searching lecture chunks and asking Groq..."):
                # RAG Step 1 (Retrieval): Find the top 3 chunks matching the question's keywords
                relevant_chunks = retrieve_relevant_chunks(
                    st.session_state["chunks"], 
                    user_question, 
                    top_k=3
                )

                # RAG Step 2 & 3 (Augmentation & Generation): Send to Groq
                answer = generate_answer(
                    relevant_chunks, 
                    user_question, 
                    api_key
                )

            # UI Element 5: Area to display the LLM's answer
            st.markdown("### Answer")
            st.info(answer)

        except Exception as e:
            st.error(f"Error generating answer: {e}")
