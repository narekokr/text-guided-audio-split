import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000"

# --- Page Configuration ---
st.set_page_config(
    page_title="Audio Assistant",
    page_icon="🎵",
    layout="centered"
)

# --- Custom CSS for Chat Alignment and Styling ---
st.markdown("""
    <style>
        /* Main layout container */
        .main .block-container {
            max-width: 900px;
            padding: 1rem 1.5rem 3rem 1.5rem;
        }

        /* Expander for the uploader */
        .st-expander {
            border-color: #E5E7EB !important;
            border-radius: 12px !important;
        }

        /* This is the container for the chat history */
        .chat-container {
            display: flex;             /* Enables Flexbox */
            flex-direction: column;    /* Stacks messages vertically */
            gap: 1.2rem;               /* FIXED: Adds margin between bubbles */
            margin-bottom: 2rem;
        }

        /* Common style for all chat messages */
        .chat-message {
            padding: 0.9rem 1.3rem;
            border-radius: 1.1rem;
            max-width: 75%;
            word-wrap: break-word;
        }

        .user-msg {
            background-color: #2563EB;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .assistant-msg {
            background-color: #E5E7EB;
            color: black;
            margin-right: auto;
            text-align: left;
        }

        /* Download button styling */
        .download-btn {
            display: inline-block;
            padding: 0.6em 1.2em;
            background-color: #6366F1;
            color: white !important;
            font-weight: 600;
            border: none;
            border-radius: 12px;
            text-decoration: none !important;
        }
        .stMarkdown:has(.chat-message) {
            margin-bottom: 2em;
        }
    </style>
""", unsafe_allow_html=True)


# --- Session State Management ---
def reset_session_for_new_file():
    """Clears previous file data and chat history to start fresh, as if no file was ever uploaded."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.file_uploaded = False  # Crucially set to False
    st.session_state.chat_history = []
    st.session_state.last_stems = []
    st.session_state.last_remix = None
    st.session_state.processing_chat = False
    st.session_state.current_filename = None  # Crucially set to None


# Initialize session state if it doesn't exist
if "session_id" not in st.session_state:
    reset_session_for_new_file()  # Use the function to initialize all states

# --- Main Page Layout ---
st.title("🎵 Audio Assistant")
if st.session_state.current_filename:
    st.caption(f"**Now chatting about:** `{st.session_state.current_filename}`")
else:
    st.caption("Upload an audio file below to get started.")

with st.expander("Upload a New Audio File", expanded=not st.session_state.file_uploaded):
    uploaded_file = st.file_uploader(
        "Upload your audio file (uploading a new one will start a new chat)",
        type=["mp3", "wav", "m4a"],
        key=f"uploader_{st.session_state.session_id}"
    )

    if st.button("Process Audio") and uploaded_file is not None:
        # Before processing new audio, perform a full reset (frontend and backend)
        try:
            requests.post(f"{API_URL}/reset", json={"session_id": st.session_state.session_id})
        except requests.exceptions.ConnectionError:
            st.warning("Could not connect to API to reset previous session, proceeding with new upload.")

        reset_session_for_new_file()  # Ensure a clean slate for the new file
        st.session_state.current_filename = uploaded_file.name

        with st.spinner(f"Processing {uploaded_file.name}..."):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            data = {"session_id": st.session_state.session_id}
            try:
                response = requests.post(f"{API_URL}/upload", files=files, data=data)
                if response.status_code == 200:
                    st.session_state.file_uploaded = True
                    st.success("✅ File ready! You can now start chatting below.")
                    st.rerun()
                else:
                    st.error(f"❌ Upload failed (Code: {response.status_code})")
                    reset_session_for_new_file()  # Reset if upload fails
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Connection Error. Is the API server running at {API_URL}?")
                reset_session_for_new_file()  # Reset if connection fails

# Add a Reset Chat Button
# This button is available whenever a file has been uploaded (i.e., not in initial state)
# or if there's any chat history from a previous interaction.
if st.session_state.file_uploaded or st.session_state.chat_history:
    if st.button("🔄 Start New Session", help="Clears all chat history and allows uploading a new audio file."):
        # Call the backend /reset endpoint first
        try:
            reset_response = requests.post(f"{API_URL}/reset", json={"session_id": st.session_state.session_id})
            if reset_response.status_code == 200:
                st.success("Session fully reset!")
            else:
                st.error(f"Failed to reset backend session (Code: {reset_response.status_code}).")
        except requests.exceptions.ConnectionError:
            st.error(f"Connection Error: Could not connect to the API to reset the session at {API_URL}.")

        # Always reset frontend state completely, regardless of backend reset success
        reset_session_for_new_file()
        st.rerun()  # Rerun to reflect the fully reset state

# Display chat interface only if a file has been successfully uploaded
if st.session_state.file_uploaded:
    st.markdown("---")
    if not st.session_state.chat_history and not st.session_state.processing_chat:
        st.info("Your audio is ready. Ask the assistant something, like 'separate vocals and bass'.")

    # This container is where the chat messages will be displayed
    with st.container():
        # We inject the HTML structure that our CSS will style
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            role = msg.get("role")
            content = msg.get("content")
            css_class = "user-msg" if role == "user" else "assistant-msg"
            st.markdown(f'<div class="chat-message {css_class}">{content}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.last_stems:
        st.markdown("### 🎧 Separated Stems")
        with st.container(border=True):
            for stem in st.session_state.last_stems:
                name = stem.get("name", "Unnamed Stem")
                url = stem.get("file_url")
                if url:
                    st.markdown(f"**{name.capitalize()}**")
                    st.audio(f"{API_URL}{url}", format="audio/wav")
                    st.markdown(f"""
                        <a href="{API_URL}{url}" download="{name}.wav" class="download-btn">
                            ⬇️ Download {name}.wav
                        </a><br><br>
                    """, unsafe_allow_html=True)

    # Display Remix Output
    if st.session_state.last_remix:
        st.markdown("### 🎛️ Remixed Audio")
        with st.container(border=True):
            remix_url = st.session_state.last_remix.get("file_url")
            remix_name = st.session_state.last_remix.get("name", "Remix")
            if remix_url:
                st.markdown(f"**{remix_name.capitalize()}**")
                st.audio(f"{API_URL}{remix_url}", format="audio/wav")
                st.markdown(f"""
                    <a href="{API_URL}{remix_url}" download="{remix_name}.wav" class="download-btn">
                        ⬇️ Download {remix_name}.wav
                    </a><br><br>
                """, unsafe_allow_html=True)

    st.markdown("---")
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your Message:",
            placeholder="e.g., Isolate the drums",
            key="user_input",
            disabled=st.session_state.processing_chat  # Disable if processing
        )
        submitted = st.form_submit_button(
            "Send",
            disabled=st.session_state.processing_chat  # Disable if processing
        )

    if submitted and user_input.strip():
        # Immediately add user message to history and set processing flag
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.processing_chat = True
        st.rerun()  # Rerun to show the user message and disable inputs

    # This block will run on the rerun after the user message is displayed
    if st.session_state.processing_chat and st.session_state.chat_history and st.session_state.chat_history[-1][
        "role"] == "user":
        # Only proceed if the last message was from the user and we are processing
        user_message_to_send = st.session_state.chat_history[-1]["content"]
        payload = {"session_id": st.session_state.session_id, "message": user_message_to_send}

        with st.spinner("Assistant thinking..."):  # Show spinner
            try:
                chat_response = requests.post(f"{API_URL}/chat", json=payload)
                if chat_response.status_code == 200:
                    result = chat_response.json()
                    st.session_state.chat_history = result.get("history", [])
                    st.session_state.last_stems = result.get("stems", [])
                    st.session_state.last_remix = result.get("remix")
                else:
                    st.error(f"❌ Assistant failed to respond (Code: {chat_response.status_code}).")
                    # Remove the last user message if the assistant failed to respond
                    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                        st.session_state.chat_history.pop()  # Remove user message from history
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Connection Error. Could not contact the assistant at {API_URL}.")
                # Remove the last user message if there was a connection error
                if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                    st.session_state.chat_history.pop()  # Remove user message from history
            finally:
                st.session_state.processing_chat = False  # Always unset the flag
                st.rerun()  # Rerun to show assistant's response and re-enable inputs