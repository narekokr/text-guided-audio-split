import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000"  # TODO: Move to config/secrets

st.set_page_config(page_title="Audio Assistant", page_icon="🎵", layout="wide")

# Session management
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🎵 Audio Assistant — Upload Then Chat")
st.markdown("Session resets on refresh. Upload your audio and chat in one run.")

# --- Upload Section ---
if not st.session_state.file_uploaded:
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
    if uploaded_file:
        if st.button("Upload & Start"):
            with st.spinner("Uploading..."):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                data = {"session_id": st.session_state.session_id}
                try:
                    response = requests.post(f"{API_URL}/upload", files=files, data=data)
                    if response.status_code == 200:
                        st.success("✅ File uploaded and converted successfully!")
                        st.session_state.file_uploaded = True
                    else:
                        error_msg = response.text if response.text else "Upload failed."
                        st.error(f"❌ {error_msg}")
                except Exception as e:
                    st.error(f"❌ Upload error: {e}")
    st.markdown("---")
else:
    st.info("File already uploaded. You can chat with the assistant or upload a new file.")
    if st.button("Reset Session"):
        st.session_state.file_uploaded = False
        st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.experimental_rerun()

# --- Chat Section ---
if st.session_state.file_uploaded:
    st.markdown("### 💬 Chat with the Assistant")
    prompt = st.text_input("Your prompt", placeholder="e.g. extract vocals and drums", key="prompt_input")

    if st.button("Send Prompt", disabled=not prompt.strip()):
        with st.spinner("Assistant is processing..."):
            payload = {
                "session_id": st.session_state.session_id,
                "message": prompt
            }
            try:
                chat_response = requests.post(f"{API_URL}/chat", json=payload)
                if chat_response.status_code == 200:
                    result = chat_response.json()
                    st.session_state.chat_history.append(("user", prompt))
                    st.session_state.chat_history.append(("assistant", result.get("reply", "")))

                    # Display chat history
                    for role, msg in st.session_state.chat_history:
                        if role == "user":
                            st.markdown(f"**You:** {msg}")
                        else:
                            st.markdown(f"**Assistant:** {msg}")

                    # Display stems
                    for stem in result.get("stems", []):
                        name = stem["name"]
                        url = stem["file_url"]
                        st.markdown(f"**🎧 {name.capitalize()}**")
                        st.audio(f"{API_URL}{url}", format="audio/wav")
                        st.markdown(
                            f'<a href="{API_URL}{url}" download="{name}.wav" class="download-btn">'
                            f"⬇️ Download {name}.wav</a>",
                            unsafe_allow_html=True)
                else:
                    try:
                        st.error(f"❌ Assistant failed to respond: {chat_response.json().get('detail', chat_response.text)}")
                    except Exception:
                        st.error("❌ Assistant failed to respond.")
            except Exception as e:
                st.error(f"❌ Error contacting backend: {e}")

    # Show chat history even after reload
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")

# --- Styling ---
st.markdown("""
    <style>
    .download-btn {
        display: inline-block;
        padding: 0.6em 1.2em;
        background-color: #6366F1;
        color: white !important;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        text-decoration: none !important;
        font-size: 16px;
        transition: background-color 0.2s ease;
        margin-bottom: 1em;
    }
    .download-btn:hover {
        background-color: #4F46E5;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)