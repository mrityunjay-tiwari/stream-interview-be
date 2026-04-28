import os
import json
import uuid
import asyncio
from fastapi import FastAPI, Request
from getstream import Stream
from openai import AsyncOpenAI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from vision_agents.core import agents
from vision_agents.core.edge.types import User
from vision_agents.core.stt.events import STTTranscriptEvent
from vision_agents.core.turn_detection.events import TurnStartedEvent, TurnEndedEvent
from vision_agents.plugins import getstream, deepgram, openrouter, smart_turn

MIN_ANSWER_WORDS = 3
TURN_DEBOUNCE_SECONDS = 1.0
PRE_SPEAK_DELAY_SECONDS = 0.4
SECTION_WRAP_UP_BUFFER_SECONDS = 90

SECTION_LABELS = {
    "projects": "Projects",
    "frontend": "Frontend",
    "backend": "Backend",
    "behavioral": "Behavioral",
    "dsa": "DSA",
    "system_design": "System Design",
    "ml_fundamentals": "ML Fundamentals",
}

SECTION_PROMPT_HINTS = {
    "projects": "Project walkthroughs, implementation ownership, execution decisions, and practical tradeoffs.",
    "frontend": "React, browser behavior, UI architecture, rendering, state, performance, accessibility, and debugging.",
    "backend": "APIs, databases, services, reliability, debugging, distributed systems basics, and tradeoffs.",
    "behavioral": "Communication, conflict resolution, ownership, leadership, teamwork, ambiguity, and decision-making.",
    "dsa": "Problem-solving, algorithmic reasoning, time and space complexity, tradeoffs, and correctness.",
    "system_design": "Architecture, scaling, system boundaries, data flow, reliability, tradeoffs, and evolution over time.",
    "ml_fundamentals": "Modeling intuition, training, evaluation, error analysis, deployment thinking, and practical ML tradeoffs.",
}

sessions = {}
active_agents = {}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

STREAM_API_KEY = os.getenv("STREAM_API_KEY")
STREAM_API_SECRET = os.getenv("STREAM_API_SECRET")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

stream_client = Stream(
    api_key=STREAM_API_KEY,
    api_secret=STREAM_API_SECRET,
)

llm_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def now_ts() -> float:
    return asyncio.get_running_loop().time()


def extract_json(raw: str):
    raw = (raw or "").strip()

    if raw.startswith("```json"):
        raw = raw[len("```json"):].strip()
    elif raw.startswith("```"):
        raw = raw[len("```"):].strip()

    if raw.endswith("```"):
        raw = raw[:-3].strip()

    return json.loads(raw)


def safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def is_repeat_request(answer: str) -> bool:
    normalized = answer.lower().strip()

    repeat_phrases = [
        "please repeat",
        "can you repeat",
        "repeat the question",
        "say that again",
        "could you repeat",
        "can you say that again",
        "i'm sorry",
        "sorry, can you repeat",
        "please ask again",
    ]

    return any(phrase in normalized for phrase in repeat_phrases)


def make_section(
    section_type: str,
    duration_minutes: int,
    min_questions: int,
    max_questions: int,
    focus_topics: list[str] | None = None,
):
    duration_minutes = max(5, min(45, duration_minutes))
    min_questions = max(1, min(25, min_questions))
    max_questions = max(min_questions, min(30, max_questions))

    return {
        "type": section_type,
        "label": SECTION_LABELS[section_type],
        "duration_minutes": duration_minutes,
        "min_questions": min_questions,
        "max_questions": max_questions,
        "focus_topics": focus_topics or [],
        "started_at": None,
        "ended_at": None,
        "state": "PENDING",
        "questions_completed": 0,
        "followups_used": 0,
    }


def default_flow_for_role(role: str):
    normalized_role = role.lower()

    if "ml" in normalized_role or "ai" in normalized_role:
        return [
            make_section("projects", 10, 2, 5),
            make_section("ml_fundamentals", 15, 3, 10),
            make_section("behavioral", 10, 3, 8),
        ]

    if "backend" in normalized_role:
        return [
            make_section("projects", 10, 2, 5),
            make_section("backend", 15, 4, 14),
            make_section("behavioral", 10, 3, 8),
        ]

    if "system design" in normalized_role:
        return [
            make_section("projects", 10, 2, 5),
            make_section("system_design", 20, 2, 8),
            make_section("behavioral", 10, 3, 8),
        ]

    return [
        make_section("projects", 10, 2, 5),
        make_section("frontend", 15, 4, 15),
        make_section("behavioral", 10, 3, 8),
    ]


def normalize_flow(raw_flow, role: str):
    if not isinstance(raw_flow, list) or not raw_flow:
        return default_flow_for_role(role)

    normalized = []

    for raw_section in raw_flow:
        if not isinstance(raw_section, dict):
            continue

        section_type = raw_section.get("type")
        if section_type not in SECTION_LABELS:
            continue

        normalized.append(
            make_section(
                section_type=section_type,
                duration_minutes=safe_int(raw_section.get("duration_minutes"), 15),
                min_questions=safe_int(raw_section.get("min_questions"), 3),
                max_questions=safe_int(raw_section.get("max_questions"), 10),
                focus_topics=raw_section.get("focus_topics") or [],
            )
        )

    if not normalized:
        return default_flow_for_role(role)

    return normalized


def get_current_section(session: dict):
    flow = session.get("flow", [])
    index = session.get("current_section_index", 0)

    if 0 <= index < len(flow):
        return flow[index]

    return None


def get_current_section_elapsed_seconds(session: dict):
    section = get_current_section(session)
    if not section or section.get("started_at") is None:
        return 0

    return max(0, int(now_ts() - section["started_at"]))


def start_section(session: dict, index: int):
    flow = session["flow"]

    if index >= len(flow):
        return None

    session["current_section_index"] = index
    section = flow[index]

    if session.get("session_started_at") is None:
        session["session_started_at"] = now_ts()

    if section["started_at"] is None:
        section["started_at"] = now_ts()

    section["state"] = "ACTIVE"
    return section


def advance_to_next_section(session: dict):
    current_section = get_current_section(session)

    if current_section:
        current_section["ended_at"] = now_ts()
        current_section["state"] = "DONE"

    next_index = session["current_section_index"] + 1

    if next_index >= len(session["flow"]):
        return None

    return start_section(session, next_index)


def build_session_status(session: dict):
    section = get_current_section(session)

    if not section:
        return {
            "currentSection": None,
            "currentSectionLabel": None,
            "currentSectionIndex": 0,
            "totalSections": len(session.get("flow", [])),
            "sectionState": "DONE",
            "elapsedSeconds": 0,
            "durationSeconds": 0,
            "questionsCompleted": 0,
        }

    return {
        "currentSection": section["type"],
        "currentSectionLabel": section["label"],
        "currentSectionIndex": session["current_section_index"],
        "totalSections": len(session.get("flow", [])),
        "sectionState": section["state"],
        "elapsedSeconds": get_current_section_elapsed_seconds(session),
        "durationSeconds": section["duration_minutes"] * 60,
        "questionsCompleted": section["questions_completed"],
    }


def build_transcript(segments: list[dict]):
    return "\n\n".join(
        [
            f"Q{index + 1}: {segment['question']}\nA{index + 1}: {segment['answer']}"
            for index, segment in enumerate(segments)
        ]
    )


def build_section_transcript(segments: list[dict], section_type: str):
    filtered = [segment for segment in segments if segment.get("section_type") == section_type]
    return build_transcript(filtered)


def should_transition_sections(session: dict, section: dict):
    elapsed_seconds = get_current_section_elapsed_seconds(session)
    duration_seconds = section["duration_minutes"] * 60
    min_questions = section["min_questions"]
    max_questions = section["max_questions"]
    questions_completed = section["questions_completed"]

    if questions_completed >= max_questions:
        section["state"] = "TRANSITIONING"
        return True

    if elapsed_seconds >= duration_seconds and questions_completed >= min_questions:
        section["state"] = "TRANSITIONING"
        return True

    if elapsed_seconds >= duration_seconds and questions_completed < min_questions:
        section["state"] = "WRAP_UP"
        return False

    wrap_up_threshold = max(0, duration_seconds - SECTION_WRAP_UP_BUFFER_SECONDS)
    if elapsed_seconds >= wrap_up_threshold and questions_completed >= min_questions:
        section["state"] = "WRAP_UP"
    else:
        section["state"] = "ACTIVE"

    return False


async def parse_turn_response(response_content: str, fallback_question: str):
    try:
        data = extract_json(response_content)
        spoken_text = (data.get("spoken_text") or "").strip()
        question_text = (data.get("question_text") or "").strip()

        if not spoken_text and not question_text:
            raise ValueError("Empty turn response")

        if not spoken_text:
            spoken_text = question_text
        if not question_text:
            question_text = spoken_text

        return {
            "spoken_text": spoken_text,
            "question_text": question_text,
        }
    except Exception:
        return {
            "spoken_text": fallback_question,
            "question_text": fallback_question,
        }


async def generate_opening_turn(session: dict, section: dict):
    prompt = f"""
You are a professional mock interviewer.
Role:
{session['role']}
Target seniority:
{session['seniority']}
Current section:
{section['label']}
Section guidance:
{SECTION_PROMPT_HINTS[section['type']]}
Ask the opening question for this section.
Rules:
- Ask exactly one question.
- Keep it natural and interview-like.
- If this is the projects section, a short intro/background question is acceptable.
- Do not ask multiple questions.
- Return STRICT JSON only.
{{
  "spoken_text": "what the interviewer should say out loud",
  "question_text": "the exact question being asked"
}}
"""

    fallback_question = "Please introduce yourself and tell me about a recent project relevant to this role."

    try:
        response = await llm_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=180,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty opening turn response")

        return await parse_turn_response(content, fallback_question)
    except Exception as error:
        print("Opening Turn Generation Error:", error)
        return {
            "spoken_text": fallback_question,
            "question_text": fallback_question,
        }


async def generate_next_turn(
    session: dict,
    section: dict,
    current_question: str,
    answer: str,
):
    full_transcript = build_transcript(session["segments"])
    section_transcript = build_section_transcript(session["segments"], section["type"])

    wrap_up_instruction = (
        "You are close to the section boundary. Ask one concise final question in this section."
        if section["state"] == "WRAP_UP"
        else "Continue the section naturally."
    )

    prompt = f"""
You are a professional mock interviewer.
Role:
{session['role']}
Target seniority:
{session['seniority']}
Current section:
{section['label']}
Section guidance:
{SECTION_PROMPT_HINTS[section['type']]}
All interview transcript so far:
{full_transcript}
Current section transcript:
{section_transcript}
Most recent question:
{current_question}
Most recent answer:
{answer}
Instructions:
- Use the candidate's most recent answer to decide the next question.
- If the answer is shallow, probe deeper on the same topic.
- If the answer is sufficient, move forward within the same section.
- Do not ask multiple questions in one turn.
- Keep it natural and interview-like.
- {wrap_up_instruction}
- Return STRICT JSON only.
{{
  "spoken_text": "what the interviewer should say out loud",
  "question_text": "the exact question being asked"
}}
"""

    fallback_question = "Can you go one level deeper on that and explain your decision-making?"

    try:
        response = await llm_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=180,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty next-turn response")

        return await parse_turn_response(content, fallback_question)
    except Exception as error:
        print("Next Turn Generation Error:", error)
        return {
            "spoken_text": fallback_question,
            "question_text": fallback_question,
        }


async def generate_transition_turn(
    session: dict,
    previous_section: dict,
    next_section: dict,
):
    full_transcript = build_transcript(session["segments"])

    prompt = f"""
You are a professional mock interviewer.
Role:
{session['role']}
Target seniority:
{session['seniority']}
Completed section:
{previous_section['label']}
Next section:
{next_section['label']}
Next section guidance:
{SECTION_PROMPT_HINTS[next_section['type']]}
Interview transcript so far:
{full_transcript}
Generate the transition into the next section.
Rules:
- Start with one short transition sentence.
- Then ask exactly one opening question for the next section.
- Keep it natural and interview-like.
- Do not ask multiple questions beyond the single opening question.
- Return STRICT JSON only.
{{
  "spoken_text": "full spoken transition plus question",
  "question_text": "only the actual new question being asked"
}}
"""

    fallback_question = f"Let's move to the {next_section['label']} round. What would you say is the most important concept in this area for your level?"

    try:
        response = await llm_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=180,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty transition turn response")

        return await parse_turn_response(content, fallback_question)
    except Exception as error:
        print("Transition Turn Generation Error:", error)
        return {
            "spoken_text": fallback_question,
            "question_text": fallback_question.split(". ", 1)[-1].strip(),
        }


async def create_agent(role: str):
    instructions = f"""
    You are a professional mock interviewer for a {role}.
    Ask exactly one interview question at a time.
    Keep questions clear and natural.
    Do not ask multiple questions in one turn.
    If the candidate asks for a repeat, repeat only the same question.
    Do not give long explanations unless explicitly asked.
    """

    agent = agents.Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Interview Coach", id="agent"),
        instructions=instructions,
        llm=openrouter.LLM(
            model="openai/gpt-4o-mini"
        ),
        stt=deepgram.STT(),
        tts=deepgram.TTS(),
        turn_detection=smart_turn.TurnDetection(),
    )

    await agent.turn_detection.warmup()
    return agent


async def evaluate_segment(
    call_id: str,
    question: str,
    answer: str,
    role: str,
    seniority: str,
    section_type: str,
):
    try:
        if call_id not in sessions:
            return None

        prompt = f"""
You are evaluating a mock interview answer.
Target role:
{role}
Target seniority:
{seniority}
Section:
{SECTION_LABELS.get(section_type, section_type)}
Question:
{question}
Answer:
{answer}
Return STRICT JSON only:
{{
  "short_feedback": "1 sentence feedback",
  "score": 0
}}
"""

        response = await llm_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=80,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty evaluation response from model")

        raw = content.strip()
        print("\nEVALUATION RAW:", raw)

        data = extract_json(raw)

        feedback = {
            "short_feedback": data.get("short_feedback", ""),
            "score": data.get("score", 0),
        }

        if call_id in sessions:
            sessions[call_id]["latest_feedback"] = feedback
            sessions[call_id]["feedback_history"].append(
                {
                    "question": question,
                    "section_type": section_type,
                    "score": feedback["score"],
                    "short_feedback": feedback["short_feedback"],
                }
            )

        print(f"\n[LOG] Feedback generated for call {call_id}:")
        print(json.dumps(feedback, indent=2))

        return feedback

    except Exception as error:
        print("Evaluation Error:", error)
        return None


async def safe_say(agent: agents.Agent, session: dict, call_id: str, agent_instance_id: str, text: str):
    await asyncio.sleep(PRE_SPEAK_DELAY_SECONDS)

    if session.get("interview_ended"):
        return

    if session.get("user_speaking"):
        print(f"[AGENT {agent_instance_id}] skipping speech because user started speaking again in call {call_id}")
        return

    if not text or not text.strip():
        return

    await agent.say(text.strip())


async def join_call(agent: agents.Agent, call_type: str, call_id: str, agent_instance_id: str):
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)

    session = sessions[call_id]

    if session.get("agent_joined"):
        print(f"[AGENT {agent_instance_id}] duplicate join prevented for call {call_id}")
        return

    session["agent_joined"] = True
    first_section = start_section(session, 0)

    @agent.events.subscribe
    async def on_transcript(event: STTTranscriptEvent):
        if session.get("interview_ended"):
            return
        session["current_answer_buffer"].append(event.text)

    @agent.events.subscribe
    async def on_turn_started(event: TurnStartedEvent):
        if event.participant and event.participant.user_id == "agent":
            return

        current_time = now_ts()
        last_started = session.get("last_turn_started_at", 0.0)

        if current_time - last_started < TURN_DEBOUNCE_SECONDS:
            return

        session["last_turn_started_at"] = current_time
        session["user_speaking"] = True
        print(f"[AGENT {agent_instance_id}] user started speaking in call {call_id}")

    @agent.events.subscribe
    async def on_turn_ended(event: TurnEndedEvent):
        if event.participant and event.participant.user_id == "agent":
            return

        if session.get("interview_ended"):
            return

        current_time = now_ts()
        last_ended = session.get("last_turn_ended_at", 0.0)

        if current_time - last_ended < TURN_DEBOUNCE_SECONDS:
            print(f"[AGENT {agent_instance_id}] duplicate turn end debounced for call {call_id}")
            return

        session["last_turn_ended_at"] = current_time
        session["user_speaking"] = False

        if session.get("processing_turn"):
            print(f"[AGENT {agent_instance_id}] duplicate turn end ignored for call {call_id}")
            return

        session["processing_turn"] = True

        try:
            current_section = get_current_section(session)
            if not current_section:
                session["interview_ended"] = True
                return

            print(f"[AGENT {agent_instance_id}] user finished speaking in call {call_id}")

            full_answer = " ".join(session["current_answer_buffer"]).strip()
            session["current_answer_buffer"] = []

            if not full_answer:
                return

            current_question = session.get("current_question")
            if not current_question:
                print(f"[AGENT {agent_instance_id}] no current question set, ignoring answer")
                return

            if is_repeat_request(full_answer):
                print(f"[AGENT {agent_instance_id}] user asked to repeat the question in call {call_id}")
                session["current_answer_buffer"] = []
                session["last_processed_answer"] = None
                await safe_say(agent, session, call_id, agent_instance_id, current_question)
                return

            if len(full_answer.split()) < MIN_ANSWER_WORDS:
                print(f"[AGENT {agent_instance_id}] answer too short, ignoring: {full_answer!r}")
                return

            if full_answer == session.get("last_processed_answer"):
                print(f"[AGENT {agent_instance_id}] duplicate answer ignored")
                return

            session["last_processed_answer"] = full_answer

            segment = {
                "question": current_question,
                "answer": full_answer,
                "section_type": current_section["type"],
            }

            session["segments"].append(segment)
            current_section["questions_completed"] += 1

            print(f"[AGENT {agent_instance_id}] segments for call {call_id}:")
            print(json.dumps(session["segments"], indent=2))

            asyncio.create_task(
                evaluate_segment(
                    call_id,
                    current_question,
                    full_answer,
                    session["role"],
                    session["seniority"],
                    current_section["type"],
                )
            )

            if should_transition_sections(session, current_section):
                previous_section = dict(current_section)
                next_section = advance_to_next_section(session)

                if next_section is None:
                    session["interview_ended"] = True
                    await safe_say(
                        agent,
                        session,
                        call_id,
                        agent_instance_id,
                        "That concludes the interview. Thank you for taking part.",
                    )
                    return

                turn = await generate_transition_turn(session, previous_section, next_section)
            else:
                turn = await generate_next_turn(
                    session,
                    current_section,
                    current_question,
                    full_answer,
                )

            if session.get("interview_ended"):
                return

            session["current_question"] = turn["question_text"]
            session["current_answer_buffer"] = []
            session["last_processed_answer"] = None

            print(
                f"[AGENT {agent_instance_id}] next question for call {call_id}: "
                f"{turn['question_text']}"
            )
            await safe_say(agent, session, call_id, agent_instance_id, turn["spoken_text"])

        except Exception as error:
            print(f"[AGENT {agent_instance_id}] turn processing error for call {call_id}: {error}")
        finally:
            session["processing_turn"] = False

    try:
        async with agent.join(call):
            print(f"[AGENT {agent_instance_id}] joined call {call_id}")

            session["current_answer_buffer"] = []
            session["last_processed_answer"] = None

            opening_turn = await generate_opening_turn(session, first_section)
            session["current_question"] = opening_turn["question_text"]

            print(f"[AGENT {agent_instance_id}] about to say intro question")
            await safe_say(
                agent,
                session,
                call_id,
                agent_instance_id,
                opening_turn["spoken_text"],
            )
            print(f"[AGENT {agent_instance_id}] intro question said")

            while not session.get("interview_ended"):
                await asyncio.sleep(1)

    finally:
        if call_id in sessions:
            sessions[call_id]["agent_joined"] = False
            sessions[call_id]["processing_turn"] = False
            sessions[call_id]["interview_ended"] = True
            sessions[call_id]["user_speaking"] = False

        active_agents.pop(call_id, None)
        print(f"[AGENT {agent_instance_id}] final cleanup complete for call {call_id}")


async def main_agent(call_type: str, call_id: str, agent_instance_id: str):
    session = sessions[call_id]
    agent = await create_agent(session["role"])
    await join_call(agent, call_type, call_id, agent_instance_id)


@app.get("/")
async def health():
    return {"status": "running", "message": "healthy"}


@app.get("/segments/{call_id}")
async def get_segments(call_id: str):
    session = sessions.get(call_id)
    if not session:
        return {"segments": []}
    return {"segments": session.get("segments", [])}


@app.get("/session-status/{call_id}")
async def get_session_status(call_id: str):
    session = sessions.get(call_id)

    if not session:
        return {
            "currentSection": None,
            "currentSectionLabel": None,
            "currentSectionIndex": 0,
            "totalSections": 0,
            "sectionState": "DONE",
            "elapsedSeconds": 0,
            "durationSeconds": 0,
            "questionsCompleted": 0,
        }

    return build_session_status(session)


@app.post("/create-token")
async def create_token(user_id: str):
    token = stream_client.create_token(user_id)
    return {
        "apiKey": STREAM_API_KEY,
        "token": token,
        "userId": user_id,
    }


@app.post("/create-session")
async def create_session(request: Request):
    data = {}

    try:
        raw = await request.body()
        if raw:
            data = json.loads(raw.decode("utf-8"))
    except Exception:
        data = {}

    role = data.get("role", "React Developer")
    seniority = data.get("seniority", "SDE1")
    flow = normalize_flow(data.get("flow"), role)

    call_id = str(uuid.uuid4())

    sessions[call_id] = {
        "role": role,
        "seniority": seniority,
        "flow": flow,
        "current_section_index": 0,
        "session_started_at": None,
        "latest_feedback": None,
        "feedback_history": [],
        "segments": [],
        "current_question": None,
        "current_answer_buffer": [],
        "agent_started": False,
        "agent_joined": False,
        "processing_turn": False,
        "interview_ended": False,
        "agent_instance_id": None,
        "last_processed_answer": None,
        "status": "active",
        "last_turn_started_at": 0.0,
        "last_turn_ended_at": 0.0,
        "user_speaking": False,
    }

    return {
        "call_id": call_id,
        "session_config": {
            "role": role,
            "seniority": seniority,
            "flow": flow,
        },
    }


@app.get("/latest-feedback/{call_id}")
async def get_latest_feedback(call_id: str):
    session = sessions.get(call_id)

    if not session:
        return {"feedback": None}

    return {"feedback": session.get("latest_feedback")}


@app.post("/end-call")
async def end_call(data: dict):
    call_id = data.get("call_id")

    if call_id in sessions:
        sessions[call_id]["status"] = "ended"
        sessions[call_id]["interview_ended"] = True
        return {"message": "Call ended successfully"}

    return {"error": "Session not found"}


@app.post("/start-agent")
async def start_agent(data: dict):
    call_id = data.get("call_id")

    if not call_id:
        return {"error": "call_id required"}

    if call_id not in sessions:
        return {"error": "session not initialized"}

    session = sessions[call_id]

    if session.get("agent_started"):
        print(f"[WARN] Duplicate start-agent ignored for call {call_id}")
        return {"status": "already started"}

    if call_id in active_agents:
        print(f"[WARN] Active agent already exists for call {call_id}")
        return {"status": "already running"}

    agent_instance_id = str(uuid.uuid4())[:8]
    session["agent_started"] = True
    session["agent_joined"] = False
    session["processing_turn"] = False
    session["interview_ended"] = False
    session["agent_instance_id"] = agent_instance_id
    session["current_answer_buffer"] = []
    session["last_processed_answer"] = None
    session["last_turn_started_at"] = 0.0
    session["last_turn_ended_at"] = 0.0
    session["user_speaking"] = False

    async def runner():
        try:
            print(f"[AGENT {agent_instance_id}] starting for call {call_id}")
            await main_agent("default", call_id, agent_instance_id)
        finally:
            active_agents.pop(call_id, None)
            if call_id in sessions:
                sessions[call_id]["agent_started"] = False
                sessions[call_id]["agent_joined"] = False
                sessions[call_id]["processing_turn"] = False
                sessions[call_id]["user_speaking"] = False
            print(f"[AGENT {agent_instance_id}] cleanup complete for call {call_id}")

    task = asyncio.create_task(runner())
    active_agents[call_id] = task

    return {
        "status": "agent started",
        "agent_instance_id": agent_instance_id,
        "role": session["role"],
        "seniority": session["seniority"],
    }
