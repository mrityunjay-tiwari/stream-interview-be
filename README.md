# Stream Interview Backend (stream-be) 

This is the backend for the **Stream Interview** platform—a professional AI-powered mock interview tool. It leverages cutting-edge AI agents to conduct real-time interviews, provide instant technical and behavioral feedback, and facilitate high-quality video/audio communication.

---

### 🔗 Linked Repositories

- **Frontend Repository:** [https://github.com/mrityunjay-tiwari/interview-tool](https://github.com/mrityunjay-tiwari/interview-tool)

---

##  Features

- ** AI-Powered Interviewer:** Conducts professional mock interviews using advanced LLMs (like Qwen 2.5) via OpenRouter.
- ** Real-Time STT:** Uses Deepgram for high-speed Speech-to-Text during the live interview.
- **Stream Integration:** Seamlessly integrates with GetStream for high-quality video and audio calling.
- **Automated Evaluation:** Analyzes user responses in real-time, providing technical feedback and performance scores (0-10).
- **Session Management:** Organizes interview segments (question/answer pairs) for post-interview review.
- **Docker Support:** Fully containerized for easy and consistent deployment.

---

## Tech Stack

- **Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Python 3.12+)
- **AI Agent Framework:** [Vision Agents](https://github.com/vision-agents/vision-agents)
- **STT/TTS Provider:** [Deepgram](https://deepgram.com/)
- **Video/Audio Infrastructure:** [GetStream](https://getstream.io/)
- **LLM Endpoint:** [OpenRouter](https://openrouter.ai/) (using Qwen/OpenAI models)
- **Dependency Management:** [uv](https://github.com/astral-sh/uv)
- **Containerization:** Docker

---

## Project Structure

```text
├── agent.py         # Core AI agent logic, event handling, and evaluation
├── server.py        # Stream token creation and basic API setup
├── main.py          # Entry point (simple script)
├── Dockerfile       # Container definition
├── pyproject.toml   # Dependencies and project metadata
└── .env             # Environment variables (required)
```

---

## Getting Started

### 1. Prerequisites

Ensure you have the following installed:

- [Python 3.12+](https://www.python.org/downloads/)
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`
- [Docker](https://www.docker.com/) (for containerized deployment)

### 2. Environment Setup

Create a `.env` file in the root directory and add the following keys:

```env
STREAM_API_KEY=your_stream_key
STREAM_API_SECRET=your_stream_secret
OPENROUTER_API_KEY=your_openrouter_key
DEEPGRAM_API_KEY=your_deepgram_key
```

### 3. Installation

Install dependencies using `uv`:

```bash
uv sync
```

### 4. Running the Development Server

You can run the application servers:

```bash
# To run the agent-based server
uvicorn agent:app --reload --port 8000

# To run the stream token server
uvicorn server:app --reload --port 8001
```

---

## API Reference

| Endpoint                     | Method | Description                                                                       |
| :--------------------------- | :----- | :-------------------------------------------------------------------------------- |
| `/create-session`            | `POST` | Initializes a new interview session and returns a `call_id`.                      |
| `/start-agent`               | `POST` | Joins the AI interviewer into the specified call (requires `role` and `call_id`). |
| `/latest-feedback/{call_id}` | `GET`  | Retrieves the feedback for the most recent answer in a session.                   |
| `/segments/{call_id}`        | `GET`  | Returns all question-answer segments recorded during the interview.               |
| `/create-token`              | `POST` | Generates a Stream token for a specific user ID.                                  |
| `/health`                    | `GET`  | Health check endpoint.                                                            |

---

## 🐳 Running with Docker

1. **Build the image:**

   ```bash
   docker build -t stream-be .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 --env-file .env stream-be
   ```

---

