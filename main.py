# Very Important Note : This file may seem very large to you, but every route and every function has been independently written in a very clean order, I'm currently just lazy to refactor it and the Docker file, hence it's so! Soon getting the motivation to do it.
# This is the backend being used in production for the AI Interview Voice Agent, it is the major orchestration layer. It uses vision-agents library to create agents, deepgram (collecting money to switch to Cartesia) for STT and TTS and OpenRouter for LLM.

import os
import json
import uuid
import asyncio
from contextlib import suppress
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
PRE_SPEAK_DELAY_SECONDS = 0.25
SECTION_WRAP_UP_BUFFER_SECONDS = 90
MAX_SPOKEN_TEXT_LENGTH = 120

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
    "projects": "Project walkthroughs, implementation ownership, execution decisions, practical tradeoffs, delivery choices, and technical ownership.",
    "frontend": "React, browser behavior, UI architecture, rendering, state, performance, accessibility, testing, and debugging.",
    "backend": "APIs, databases, service design, reliability, observability, debugging, distributed systems basics, and engineering tradeoffs.",
    "behavioral": "Communication, conflict resolution, ownership, leadership, teamwork, ambiguity, prioritization, and decision-making.",
    "dsa": "Problem-solving, algorithmic reasoning, time and space complexity, edge cases, correctness, and tradeoff thinking.",
    "system_design": "Architecture, scaling, system boundaries, data flow, reliability, fault tolerance, tradeoffs, and evolution over time.",
    "ml_fundamentals": "Modeling intuition, training, evaluation, error analysis, deployment thinking, production ML tradeoffs, and practical ML reasoning.",
}

SENIORITY_PROFILES = {
    "Intern": {
        "scope": "Keep questions foundational and narrow. Prefer concrete basics, simple implementation reasoning, and project exposure over open-ended architecture.",
        "expected_depth": "A correct answer can be brief if it is relevant, clear, and technically sound.",
        "followup_style": "Use lighter follow-ups. Clarify gaps rather than aggressively probing.",
        "evaluation_bar": "Score based on correctness of fundamentals, clarity, and evidence of learning potential.",
    },
    "Junior Engineer": {
        "scope": "Focus on fundamentals, small feature ownership, debugging, and practical implementation choices.",
        "expected_depth": "Expect working knowledge of day-to-day engineering tasks with modest depth.",
        "followup_style": "Use targeted follow-ups to test understanding of implementation and reasoning.",
        "evaluation_bar": "Score based on practical understanding, relevance, and ability to explain straightforward technical choices.",
    },
    "SDE1": {
        "scope": "Focus on practical implementation details, debugging, smaller feature ownership, and solid fundamentals.",
        "expected_depth": "Expect clear understanding of common engineering concepts and direct implementation tradeoffs.",
        "followup_style": "Probe one level deeper when answers are vague or surface-level.",
        "evaluation_bar": "Score based on clarity, correctness, implementation depth, and real-world grounding.",
    },
    "SDE2": {
        "scope": "Probe implementation depth, tradeoffs, stronger ownership, reliability, and medium-complexity problem solving.",
        "expected_depth": "Expect stronger reasoning, tradeoffs, and better articulation under follow-up.",
        "followup_style": "Use sharper follow-ups to test decision-making and judgment.",
        "evaluation_bar": "Score based on technical depth, tradeoff awareness, and quality of reasoning under pressure.",
    },
    "SDE3": {
        "scope": "Expect broader ownership, stronger debugging instincts, system-level awareness, and consistent depth.",
        "expected_depth": "Answers should show independent judgment, prioritization, and stronger tradeoff awareness.",
        "followup_style": "Probe for consequences, alternatives, and decision rationale.",
        "evaluation_bar": "Score based on strong technical articulation, depth, and ownership thinking.",
    },
    "Senior Engineer": {
        "scope": "Probe architecture choices, scaling, reliability, mentoring, cross-cutting concerns, and technical leadership in execution.",
        "expected_depth": "Expect nuanced tradeoffs, debugging maturity, and clarity about system impact.",
        "followup_style": "Use direct follow-ups that test architecture judgment and leadership depth.",
        "evaluation_bar": "Score based on depth, clarity, scale-awareness, and engineering judgment.",
    },
    "Staff Engineer": {
        "scope": "Probe cross-team design thinking, platform choices, strategic tradeoffs, influence, and engineering direction.",
        "expected_depth": "Expect organization-aware technical reasoning and strong architectural clarity.",
        "followup_style": "Probe for long-term implications, boundaries, and alignment decisions.",
        "evaluation_bar": "Score based on strategic technical judgment, platform thinking, and ability to reason across systems.",
    },
    "Principal Engineer": {
        "scope": "Probe org-level technical impact, long-horizon architecture, platform strategy, and system evolution.",
        "expected_depth": "Expect exceptional clarity, deep tradeoff thinking, and strong architectural leadership.",
        "followup_style": "Probe consequences, risk management, and system evolution over time.",
        "evaluation_bar": "Score based on architectural depth, strategic influence, and mature long-term engineering judgment.",
    },
    "Senior AI Engineer": {
        "scope": "Probe applied AI system design, evaluation, reliability, inference tradeoffs, and production AI decision-making.",
        "expected_depth": "Expect strong ML/AI system reasoning plus software engineering rigor.",
        "followup_style": "Probe production constraints, reliability, evaluation quality, and system tradeoffs.",
        "evaluation_bar": "Score based on practical AI engineering depth, evaluation rigor, and production judgment.",
    },
    "Senior ML Engineer": {
        "scope": "Probe modeling, training, evaluation, deployment, monitoring, and ML tradeoffs in production environments.",
        "expected_depth": "Expect strong technical depth across model and systems sides of ML work.",
        "followup_style": "Probe assumptions, evaluation quality, and production consequences.",
        "evaluation_bar": "Score based on ML depth, practical deployment understanding, and tradeoff quality.",
    },
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


def get_seniority_profile(seniority: str):
    return SENIORITY_PROFILES.get(seniority, SENIORITY_PROFILES["SDE1"])


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


def normalize_spoken_text(text: str, fallback: str):
    value = " ".join((text or "").strip().split())
    fallback = " ".join((fallback or "").strip().split())

    if not value:
        value = fallback

    if not value:
        return ""

    if len(value) <= MAX_SPOKEN_TEXT_LENGTH:
        return value

    clipped = value[:MAX_SPOKEN_TEXT_LENGTH].rsplit(" ", 1)[0].strip()
    return clipped or fallback[:MAX_SPOKEN_TEXT_LENGTH].strip()


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


def build_intro_prompt(role: str, seniority: str):
    normalized_role = (role or "software engineer").strip()
    normalized_seniority = (seniority or "SDE1").strip()

    spoken_text = (
        f"Hi, welcome. I'll be conducting your mock interview for a "
        f"{normalized_seniority} {normalized_role} role. "
        "Before we begin, could you briefly introduce yourself and walk me through your background and recent work?"
    )

    question_text = (
        "Could you briefly introduce yourself and walk me through your background and recent work?"
    )

    return {
        "spoken_text": normalize_spoken_text(spoken_text, spoken_text),
        "question_text": normalize_spoken_text(question_text, question_text),
    }


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
            "currentQuestion": session.get("current_question"),
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
        "currentQuestion": session.get("current_question"),
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


def get_focus_topics_text(section: dict):
    topics = section.get("focus_topics") or []
    if not topics:
        return "No explicit focus topics provided."
    return ", ".join(topics)


def get_answer_signal(answer: str):
    word_count = len(answer.split())

    if word_count < 12:
        return "brief"
    if word_count < 35:
        return "moderate"
    return "detailed"


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
        spoken_text = normalize_spoken_text(
            data.get("spoken_text") or "",
            data.get("question_text") or fallback_question,
        )
        question_text = normalize_spoken_text(
            data.get("question_text") or "",
            spoken_text or fallback_question,
        )

        if not spoken_text:
            spoken_text = question_text
        if not question_text:
            question_text = spoken_text

        return {
            "spoken_text": spoken_text,
            "question_text": question_text,
        }
    except Exception:
        fallback = normalize_spoken_text(fallback_question, fallback_question)
        return {
            "spoken_text": fallback,
            "question_text": fallback,
        }


async def generate_opening_turn(session: dict, section: dict):
    profile = get_seniority_profile(session["seniority"])

    prompt = f"""
You are a professional mock interviewer.
Role:
{session['role']}
Target seniority:
{session['seniority']}
Seniority scope:
{profile['scope']}
Expected depth:
{profile['expected_depth']}
Current section:
{section['label']}
Section guidance:
{SECTION_PROMPT_HINTS[section['type']]}
Focus topics:
{get_focus_topics_text(section)}
Ask the opening question for this section.
Rules:
- Ask exactly one question.
- Keep it concise and interview-like.
- Match the difficulty to the target seniority.
- If this is the projects section, a short intro/background question is acceptable.
- Return STRICT JSON only.
{{
  "spoken_text": "short spoken question",
  "question_text": "the exact question"
}}
"""

    fallback_question = "Please tell me about a recent project relevant to this role."

    try:
        response = await llm_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=160,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty opening turn response")

        return await parse_turn_response(content, fallback_question)
    except Exception as error:
        print("Opening Turn Generation Error:", error)
        fallback = normalize_spoken_text(fallback_question, fallback_question)
        return {
            "spoken_text": fallback,
            "question_text": fallback,
        }


async def generate_next_turn(
    session: dict,
    section: dict,
    current_question: str,
    answer: str,
):
    full_transcript = build_transcript(session["segments"])
    section_transcript = build_section_transcript(session["segments"], section["type"])
    profile = get_seniority_profile(session["seniority"])
    answer_signal = get_answer_signal(answer)

    wrap_up_instruction = (
        "You are near the section boundary. Ask one short final question in this section."
        if section["state"] == "WRAP_UP"
        else "Continue the section naturally."
    )

    prompt = f"""
You are a professional mock interviewer.
Role:
{session['role']}
Target seniority:
{session['seniority']}
Seniority scope:
{profile['scope']}
Expected depth:
{profile['expected_depth']}
Follow-up style:
{profile['followup_style']}
Current section:
{section['label']}
Section guidance:
{SECTION_PROMPT_HINTS[section['type']]}
Focus topics:
{get_focus_topics_text(section)}
All interview transcript so far:
{full_transcript}
Current section transcript:
{section_transcript}
Most recent question:
{current_question}
Most recent answer:
{answer}
Answer signal:
{answer_signal}
Instructions:
- Use the candidate's most recent answer to decide the next question.
- If the answer is shallow, probe deeper on the same topic.
- If the answer is sufficient, move forward within the same section.
- Match the difficulty to the target seniority.
- Ask exactly one concise question.
- Do not ask multiple questions.
- {wrap_up_instruction}
- Return STRICT JSON only.
{{
  "spoken_text": "short spoken question",
  "question_text": "the exact question"
}}
"""

    fallback_question = "Can you go one level deeper on that and explain your decision-making?"

    try:
        response = await llm_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=160,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty next-turn response")

        return await parse_turn_response(content, fallback_question)
    except Exception as error:
        print("Next Turn Generation Error:", error)
        fallback = normalize_spoken_text(fallback_question, fallback_question)
        return {
            "spoken_text": fallback,
            "question_text": fallback,
        }


async def generate_transition_turn(
    session: dict,
    previous_section: dict,
    next_section: dict,
):
    profile = get_seniority_profile(session["seniority"])

    prompt = f"""
You are a professional mock interviewer.
Role:
{session['role']}
Target seniority:
{session['seniority']}
Seniority scope:
{profile['scope']}
Expected depth:
{profile['expected_depth']}
Completed section:
{previous_section['label']}
Next section:
{next_section['label']}
Next section guidance:
{SECTION_PROMPT_HINTS[next_section['type']]}
Focus topics:
{get_focus_topics_text(next_section)}
Generate the transition into the next section.
Rules:
- Use one short transition sentence.
- Then ask one short opening question.
- Match the difficulty to the target seniority.
- Keep the full spoken output concise.
- Return STRICT JSON only.
{{
  "spoken_text": "short transition plus short question",
  "question_text": "only the new question"
}}
"""

    fallback_spoken = f"Let's move to the {next_section['label']} round. What is the most important concept in this area for your level?"
    fallback_question = "What is the most important concept in this area for your level?"

    try:
        response = await llm_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=160,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty transition turn response")

        parsed = await parse_turn_response(content, fallback_question)
        parsed["spoken_text"] = normalize_spoken_text(parsed["spoken_text"], fallback_spoken)
        parsed["question_text"] = normalize_spoken_text(parsed["question_text"], fallback_question)
        return parsed
    except Exception as error:
        print("Transition Turn Generation Error:", error)
        return {
            "spoken_text": normalize_spoken_text(fallback_spoken, fallback_spoken),
            "question_text": normalize_spoken_text(fallback_question, fallback_question),
        }


async def create_agent(role: str):
    instructions = f"""
    You are a professional mock interviewer for a {role}.
    Ask exactly one interview question at a time.
    Keep questions clear, concise, and natural.
    If the candidate asks for a repeat, repeat only the same question.
    Do not ask multiple questions in one turn.
    """

    agent = agents.Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Interview Coach", id="agent"),
        instructions=instructions,
        llm=openrouter.LLM(model="openai/gpt-4o-mini"),
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

        profile = get_seniority_profile(seniority)

        prompt = f"""
You are evaluating a mock interview answer.
Target role:
{role}
Target seniority:
{seniority}
Section:
{SECTION_LABELS.get(section_type, section_type)}
Section guidance:
{SECTION_PROMPT_HINTS.get(section_type, section_type)}
Evaluation bar:
{profile['evaluation_bar']}
Expected depth:
{profile['expected_depth']}
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


async def cancel_active_speech(session: dict):
    task = session.get("active_speech_task")

    if task and not task.done():
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    session["active_speech_task"] = None
    session["agent_speaking"] = False


async def _perform_speech(
    agent: agents.Agent,
    session: dict,
    call_id: str,
    agent_instance_id: str,
    text: str,
    generation: int,
):
    try:
        await asyncio.sleep(PRE_SPEAK_DELAY_SECONDS)

        if session.get("interview_ended"):
            return

        if session.get("user_speaking"):
            print(f"[AGENT {agent_instance_id}] skipping speech because user started speaking again in call {call_id}")
            return

        if generation != session.get("speech_generation"):
            print(f"[AGENT {agent_instance_id}] skipping stale speech in call {call_id}")
            return

        normalized_text = normalize_spoken_text(text, text)
        if not normalized_text:
            return

        async with session["speech_lock"]:
            if session.get("interview_ended"):
                return

            if session.get("user_speaking"):
                print(f"[AGENT {agent_instance_id}] user resumed speaking before TTS in call {call_id}")
                return

            if generation != session.get("speech_generation"):
                print(f"[AGENT {agent_instance_id}] stale speech rejected before TTS in call {call_id}")
                return

            session["agent_speaking"] = True
            session["last_agent_speech_started_at"] = now_ts()

            try:
                await agent.say(normalized_text)
            finally:
                session["agent_speaking"] = False
                session["last_agent_speech_finished_at"] = now_ts()

    except asyncio.CancelledError:
        session["agent_speaking"] = False
        print(f"[AGENT {agent_instance_id}] speech task cancelled in call {call_id}")
        raise
    except Exception as error:
        session["agent_speaking"] = False
        print(f"[AGENT {agent_instance_id}] agent.say failed in call {call_id}: {error}")
    finally:
        if session.get("active_speech_task") is asyncio.current_task():
            session["active_speech_task"] = None


async def queue_speech(
    agent: agents.Agent,
    session: dict,
    call_id: str,
    agent_instance_id: str,
    text: str,
):
    session["speech_generation"] += 1
    generation = session["speech_generation"]

    await cancel_active_speech(session)

    task = asyncio.create_task(
        _perform_speech(agent, session, call_id, agent_instance_id, text, generation)
    )
    session["active_speech_task"] = task

    try:
        await task
    except asyncio.CancelledError:
        pass


async def join_call(agent: agents.Agent, call_type: str, call_id: str, agent_instance_id: str):
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)

    session = sessions[call_id]

    if session.get("agent_joined"):
        print(f"[AGENT {agent_instance_id}] duplicate join prevented for call {call_id}")
        return

    session["agent_joined"] = True

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
        session["speech_generation"] += 1
        await cancel_active_speech(session)

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
                await queue_speech(agent, session, call_id, agent_instance_id, current_question)
                return

            if len(full_answer.split()) < MIN_ANSWER_WORDS:
                print(f"[AGENT {agent_instance_id}] answer too short, ignoring: {full_answer!r}")
                return

            if full_answer == session.get("last_processed_answer"):
                print(f"[AGENT {agent_instance_id}] duplicate answer ignored")
                return

            session["last_processed_answer"] = full_answer

            if not session.get("intro_completed"):
                intro_question = session.get("current_question") or "Please introduce yourself."

                session["segments"].append(
                    {
                        "question": intro_question,
                        "answer": full_answer,
                        "section_type": "introduction",
                    }
                )

                session["intro_completed"] = True
                session["current_answer_buffer"] = []
                session["last_processed_answer"] = None

                first_section = start_section(session, 0)
                if not first_section:
                    session["interview_ended"] = True
                    await queue_speech(
                        agent,
                        session,
                        call_id,
                        agent_instance_id,
                        "That concludes the interview. Thank you for taking part.",
                    )
                    return

                opening_turn = await generate_opening_turn(session, first_section)
                session["current_question"] = opening_turn["question_text"]

                print(
                    f"[AGENT {agent_instance_id}] first section question for call {call_id}: "
                    f"{opening_turn['question_text']}"
                )
                await queue_speech(
                    agent,
                    session,
                    call_id,
                    agent_instance_id,
                    opening_turn["spoken_text"],
                )
                return

            current_section = get_current_section(session)
            if not current_section:
                session["interview_ended"] = True
                return

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
                    await queue_speech(
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
            await queue_speech(agent, session, call_id, agent_instance_id, turn["spoken_text"])

        except Exception as error:
            print(f"[AGENT {agent_instance_id}] turn processing error for call {call_id}: {error}")
        finally:
            session["processing_turn"] = False

    try:
        async with agent.join(call):
            print(f"[AGENT {agent_instance_id}] joined call {call_id}")

            session["current_answer_buffer"] = []
            session["last_processed_answer"] = None

            intro_turn = build_intro_prompt(session["role"], session["seniority"])
            session["current_question"] = intro_turn["question_text"]

            print(f"[AGENT {agent_instance_id}] about to say intro prompt")
            await queue_speech(
                agent,
                session,
                call_id,
                agent_instance_id,
                intro_turn["spoken_text"],
            )
            print(f"[AGENT {agent_instance_id}] intro prompt said")

            while not session.get("interview_ended"):
                await asyncio.sleep(1)

    finally:
        if call_id in sessions:
            sessions[call_id]["speech_generation"] += 1
            await cancel_active_speech(session)

            sessions[call_id]["agent_joined"] = False
            sessions[call_id]["processing_turn"] = False
            sessions[call_id]["interview_ended"] = True
            sessions[call_id]["user_speaking"] = False
            sessions[call_id]["agent_speaking"] = False

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
            "currentQuestion": None,
        }

    return build_session_status(session)


@app.get("/report-context/{call_id}")
async def get_report_context(call_id: str):
    session = sessions.get(call_id)

    if not session:
        return {
            "role": None,
            "seniority": None,
            "flow": [],
            "feedbackHistory": [],
        }

    return {
        "role": session.get("role"),
        "seniority": session.get("seniority"),
        "flow": session.get("flow", []),
        "feedbackHistory": session.get("feedback_history", []),
    }


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
        "agent_speaking": False,
        "speech_generation": 0,
        "active_speech_task": None,
        "speech_lock": asyncio.Lock(),
        "last_agent_speech_started_at": 0.0,
        "last_agent_speech_finished_at": 0.0,
        "intro_completed": False,
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
        sessions[call_id]["speech_generation"] += 1
        await cancel_active_speech(sessions[call_id])
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
    session["current_question"] = None
    session["last_processed_answer"] = None
    session["last_turn_started_at"] = 0.0
    session["last_turn_ended_at"] = 0.0
    session["user_speaking"] = False
    session["agent_speaking"] = False
    session["speech_generation"] = 0
    session["active_speech_task"] = None
    session["speech_lock"] = asyncio.Lock()
    session["last_agent_speech_started_at"] = 0.0
    session["last_agent_speech_finished_at"] = 0.0
    session["intro_completed"] = False

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
                sessions[call_id]["agent_speaking"] = False
            print(f"[AGENT {agent_instance_id}] cleanup complete for call {call_id}")

    task = asyncio.create_task(runner())
    active_agents[call_id] = task

    return {
        "status": "agent started",
        "agent_instance_id": agent_instance_id,
        "role": session["role"],
        "seniority": session["seniority"],
    }
