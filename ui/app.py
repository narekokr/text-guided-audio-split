import streamlit as st
import requests

API_URL = "http://localhost:8000"
st.html("""
    <style>
        .stMainBlockContainer {
            max-width:60rem;
        }
    </style>
    """
)
st.markdown("""
    <style>
        /* Make the main content container wider */
        .main .block-container {
            max-width: 1200px;
            padding-left: 3rem;
            padding-right: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

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
                name = stem["name"]
                url = stem["file_name"]

                st.markdown(f"**{name.capitalize()}**")
                st.audio(f"{API_URL}/downloads/{url}", format="audio/wav")

                st.markdown(f"""
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

                        <a href="{API_URL}/downloads/{url}" download="{name}.wav" class="download-btn">
                            ⬇️ Download {name}.wav
                        </a>
                    """, unsafe_allow_html=True)
        else:
            st.error("Something went wrong.")
