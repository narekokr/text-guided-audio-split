import streamlit as st
import requests
import base64
import io

API_URL = "http://localhost:8000"

st.title("🎵 Stream & Play Separated Stems")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
prompt = st.text_input("Describe what to extract", value="vocals, drums")

if uploaded_file and st.button("Separate"):
    with st.spinner("Processing..."):
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        data = {"prompt": prompt}

        response = requests.post(f"{API_URL}/separate", files=files, data=data)

        if response.status_code == 200:
            stems = response.json()["stems"]
            for stem in stems:
                b64 = stem["audio_base64"]
                audio_bytes = base64.b64decode(b64)
                st.audio(io.BytesIO(audio_bytes), format="audio/wav")
                st.download_button(f"Download {stem['name']}.wav", audio_bytes, file_name=f"{stem['name']}.wav", mime="audio/wav")
        else:
            st.error("Something went wrong.")
