# Text-Guided Audio Stem Separation with Demucs and Remix API

This project is a FastAPI-based backend that allows users to upload audio files, extract individual musical stems (vocals, drums, bass, other), and apply a wide range of remixing effects using natural language instructions

- **Open AI** OpenAI GPT-4 – Interprets natural language prompts and returns structured intent (e.g., stems to separate or volume adjustments).
- **Demucs** (`mdx_extra_q` model) for high-quality music source separation into standard stems: `vocals`, `drums`, `bass`, and `other`.
- **FastAPI** Backend service
- **RDBMS**  To enable multi-turn conversations, user-specific session tracking, and persistent chat history, the system integrates a PostgreSQL database via SQLModel.

> A validation layer ensures only supported stems are passed to Demucs (`vocals`, `drums`, `bass`, `other`), guarding against incorrect or unsafe prompt interpretations.

---

## 🚀 How It Works

1. **User uploads audio** and provides a **natural language prompt** (e.g. `"only separate vocals"`).
2. The prompt is interpreted by interpreter.py using OpenAI, producing response in the following sample format: { "type": "separation", "stems": ["vocals", "drums"] }
3. Demucs separates the audio into 4 standard stems.
4. Only the stems mentioned in the prompt are returned in the response.
5. Output stems are sent back via API in downloadable formats.


## 🎛️ DSP Effects (Remixing)

Apply audio effects to the mix using structured JSON or natural language instructions like:

> "Make vocals louder, add reverb to drums, and boost 3kHz by 5dB."

### Supported Effects

| Effect         | Description |
|----------------|-------------|
| **Volume Scaling** | Adjust stem loudness (e.g., `"make vocals louder"`) |
| **Reverb**     | Add reverb per stem. Range: `0.0` (none) to `1.0` (max) |
| **Pitch Shift**| Shift pitch in semitones (e.g., `+2`, `-1`) |
| **Compression**| Apply dynamic range compression. Options: `low`, `medium`, `high` |
| **EQ (Equalization)** | Boost or cut specific frequencies with `frequency`, `gain_db`, and `width` (Q) |
| **Filtering**  | Apply frequency filters: `lowpass`, `highpass`, or `bandpass` with custom cutoffs |

## 🧠 LLM-Driven Prompt Interpretation

Uses **GPT-4** to classify and interpret natural language user instructions into one of two intent types:

- **Separation**: Extract specific stems (e.g., vocals, drums)
- **Remix**: Apply DSP effects (volume, reverb, EQ, filtering, etc.)

The LLM:
- Parses user input
- Converts it into structured JSON with fields like:
  - `volumes`, `pitch_shift`, `reverb`, `compression`
  - `eq` (frequency, gain, width)
  - `filter` (type and cutoff)
- Supports flexible natural language like:
  > “Make vocals louder and add a low-pass filter at 4kHz”

---

## 🔁 Feedback Loop

Enables iterative improvements based on follow-up user feedback, such as:

> "Make vocals even louder and reduce reverb on drums"

- Detects intent using a GPT-based feedback parser
- Applies incremental changes to previously applied effects
- Tracks and reuses `last_instructions` per session for refinement

---

## ⚙️ Installation & Setup

Install dependencies and launch the FastAPI server:

```bash
pip install -r requirments.txt
uvicorn main:app --reload
