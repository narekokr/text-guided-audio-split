# Text-Guided Audio Stem Separation with Demucs & Hugging Face

This project enables natural language-based audio source separation and remixing. Users can upload an audio file and describe in simple language which components they want to isolate or adjust — such as “extract vocals and drums” or “make the bass softer” — and receive high-quality, downloadable audio stems.


- **OpenAI** OpenAI GPT-4 – Interprets natural language prompts and returns structured intent (e.g., stems to separate or volume adjustments).
- **Demucs** (`mdx_extra_q` model) for high-quality music source separation into standard stems: `vocals`, `drums`, `bass`, and `other`.
- **FastAPI** Backend service

> A validation layer ensures only supported stems are passed to Demucs (`vocals`, `drums`, `bass`, `other`), guarding against incorrect or unsafe prompt interpretations.

---

## 🚀 How It Works

1. **User uploads audio** and provides a **natural language prompt** (e.g. `"only separate vocals"`).
2. The prompt is interpreted by interpreter.py using OpenAI, producing response in the following sample format: { "type": "separation", "stems": ["vocals", "drums"] }
3. Demucs separates the audio into 4 standard stems.
4. Only the stems mentioned in the prompt are returned in the response.
5. Output stems are sent back via API in downloadable formats.
6. Remixing Flow (Experimental) - Users can also modify volume levels using prompts like:
    "Make the vocals louder, reduce the bass."
     The backend scales each separated stem using NumPy and reconstructs a remixed .wav file that matches the new mix.

---

##Prompt Example

Example input:

```text
Prompt: "Extract vocals and bass"
