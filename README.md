# Text-Guided Audio Stem Separation with Demucs and Remix API

This project is a FastAPI-based backend that allows users to upload audio files, extract individual musical stems (vocals, drums, bass, other), and apply a wide range of remixing effects using natural language instructions

- **Open AI** OpenAI GPT-4 – Interprets natural language prompts and returns structured intent (e.g., stems to separate or volume adjustments).
- **Demucs** (`mdx_extra_q` model) for high-quality music source separation into standard stems: `vocals`, `drums`, `bass`, and `other`.
- **FastAPI** Backend service
- **RDBMS**  To enable multi-turn conversations, user-specific session tracking, and persistent chat history, the system integrates a PostgreSQL database via SQLModel.

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
- **Clarification**: Handle ambiguous requests, unsupported operations, and provide helpful guidance

### Overview of the workflow

- Parses user input
- Converts it into structured JSON with fields like:
  - `volumes`, `pitch_shift`, `reverb`, `compression`
  - `eq` (frequency, gain, width)
  - `filter` (type and cutoff)
- Supports flexible natural language like:
  > “add reverb to the whole mix to make it more ambient”

## 🎵 Natural Conversation Flow
Handles diverse user requests with intelligent responses:

**Audio Processing:**
> "Separate drums and vocals"
> "Add reverb to the whole mix to make it more ambient"

**Smart Guidance:**
> "Can I get a trumpet solo?"
> *"I can only separate vocals, drums, bass, and other instruments. Trumpet would be in the other category - want to try that?"*
> "Ok give me the other stem"

**Scope Management:**
> "What time is it?"
> *"I'm focused on audio processing! Let's work with your track - want to separate stems or add some effects?"*

**Seamless Workflow:**
> "Ok, let's go back to editing. Please pitch the vocal up by 4 semitones"

- Provides educational responses for unsupported requests
- Guides users toward available features
- Maintains conversational context throughout sessions
- Handles both audio processing and general queries gracefully

---
---

## 🧠 Intelligent Intent Detection System

The LLM-powered backend uses internal intent classification to understand user requests and route them appropriately. 

### Why This Matters

**🎯 Smart Routing**  
Automatically distinguishes between separation requests ("give me vocals"), remix instructions ("make it louder"), and clarification needs ("what can you do?")

**🛡️ Robust Error Handling**  
Gracefully handles ambiguous requests, unsupported operations, and edge cases without crashes or confusing error messages

**💬 Natural User Experience**  
Provides helpful guidance instead of technical errors - when you ask for "trumpet solo," it explains the limitations and suggests alternatives

**⚙️ Modular Architecture**  
Clean separation between intent detection, audio processing, and response generation enables reliable testing and easy feature expansion

### Technical Implementation
- Multi-stage LLM classification pipeline
- Context-aware conversation state management  
- Sophisticated prompt engineering for precise parameter mapping

## 🔁 Feedback Loop
Enables iterative improvements based on follow-up user feedback, such as:

> "Make vocals even louder and reduce reverb on drums"

- Detects intent using a GPT-based feedback parser
- Applies incremental changes to previously applied effects
- Tracks and reuses `last_instructions` per session for refinement

---

## Architecture Overview


### Clean Architecture Benefits

#### **🔧 Modular Components**
- **Single Responsibility**: Each module handles one concern
- **Dependency Injection**: Clean interfaces between layers
- **Testable Units**: Isolated business logic for comprehensive testing
- **Scalable Structure**: Easy to extend without breaking existing code

#### **🎯 Separation of Concerns**
- **API Layer**: HTTP handling, response formatting
- **Business Logic**: Intent classification, audio processing workflows
- **Data Layer**: Session management, file operations, persistence
- **External Services**: LLM integration, audio processing libraries

#### **🛡️ Robust Error Handling**
- **Input Validation**: Pydantic models with type safety
- **Comprehensive Logging**: Structured logging for debugging
- **Exception Isolation**: Errors contained within module boundaries

#### **⚡ Performance Optimizations**
- **Stateless Design**: Pure functions for core logic
- **Memory Management**: Automatic cleanup of temporary files
- **Async Operations**: Non-blocking I/O for concurrent requests

### Technical Stack Integration

- **FastAPI**: Modern async web framework with automatic OpenAPI docs
- **Pydantic**: Type-safe data validation and serialization
- **SQLite**: Lightweight persistence for session management
- **NumPy/PyDub**: Optimized audio processing pipeline
- **OpenAI GPT-4**: Advanced natural language understanding

---

## ⚙️ Installation & Setup

Install dependencies and launch the FastAPI server:

```bash
pip install -r requirments.txt
uvicorn main:app --reload
