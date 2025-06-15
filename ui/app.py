import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000"

# Layout styling
st.html("""
    <style>
        .stMainBlockContainer {
            max-width:60rem;
        }
    </style>
""")
st.markdown("""
    <style>
        .main .block-container {
            max-width: 1200px;
            padding-left: 3rem;
            padding-right: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

# Generate session_id per run (resets on refresh)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

st.title("🎵 Audio Assistant — Upload Then Chat")
st.markdown("Session resets on page refresh. Upload your audio and chat in one run.")

# Upload section
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file and not st.session_state.file_uploaded:
    if st.button("Upload & Start"):
        with st.spinner("Uploading..."):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            data = {"session_id": st.session_state.session_id}

            response = requests.post(f"{API_URL}/upload", files=files, data=data)

            if response.status_code == 200:
                st.success("✅ File uploaded and converted successfully!")
                st.session_state.file_uploaded = True
            else:
                st.error("❌ Upload failed.")

# Chat section appears only after successful upload
if st.session_state.file_uploaded:
    st.markdown("---")
    st.markdown("### 💬 Chat with the Assistant")
    prompt = st.text_input("Your prompt", placeholder="e.g. extract vocals and drums")

    if st.button("Send Prompt"):
        if prompt.strip():
            with st.spinner("Assistant is processing..."):
                payload = {
                    "session_id": st.session_state.session_id,
                    "message": prompt
                }
                chat_response = requests.post(f"{API_URL}/chat", json=payload)

                if chat_response.status_code == 200:
                    result = chat_response.json()
                    st.markdown(f"**Assistant:** {result['reply']}")

                    for stem in result.get("stems", []):
                        name = stem["name"]
                        url = stem["file_url"]

                        st.markdown(f"**🎧 {name.capitalize()}**")
                        st.audio(f"{API_URL}{url}", format="audio/wav")

                        st.markdown(f"""
                            <a href="{API_URL}{url}" download="{name}.wav" class="download-btn">
                                ⬇️ Download {name}.wav
                            </a>
                            <style>
                                .download-btn {{
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
                                }}
                                .download-btn:hover {{
                                    background-color: #4F46E5;
                                    color: white;
                                }}
                            </style>
                        """, unsafe_allow_html=True)
                else:
                    st.error("❌ Assistant failed to respond.")
        else:
            st.warning("Please enter a prompt.")
