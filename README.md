# Text-Guided Audio Stem Separation with Demucs & Hugging Face

This stage of the project demonstrates how to perform **text-guided audio source separation** using:

- **Flan-T5** from Hugging Face to interpret user prompts like `"extract vocals and drums"` and convert them into target stems.
- **Demucs** (`mdx_extra_q` model) for high-quality music source separation into standard stems: `vocals`, `drums`, `bass`, and `other`.
- **FastAPI** to serve a `/separate` endpoint that accepts an audio file and a prompt, performs inference, and returns filtered audio stems in Base64 format.

> ✅ A validation layer ensures only supported stems are passed to Demucs (`vocals`, `drums`, `bass`, `other`), guarding against incorrect or unsafe prompt interpretations.

---

## 🚀 How It Works

1. **User uploads audio** and provides a **natural language prompt** (e.g. `"only separate vocals"`).
2. The prompt is passed to `Flan-T5`, which returns a **clean list of desired stems**.
3. Demucs separates the audio into 4 standard stems.
4. Only the stems mentioned in the prompt are returned in the response.
5. Output stems are encoded in Base64 and sent back via API (or can be integrated with a Streamlit frontend).

---

## 🧪 One-Shot Prompt Example

Example input:

```text
Prompt: "Extract vocals and bass only"
