import os
import asyncio
import json
from dotenv import load_dotenv
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from openai import AsyncOpenAI
from vision_agents.core import agents
from vision_agents.core.edge.types import User
from vision_agents.core.runner import Runner
from vision_agents.plugins import getstream, deepgram, cartesia, openrouter, smart_turn, ultralytics
from vision_agents.core.turn_detection.events import TurnStartedEvent, TurnEndedEvent
from vision_agents.core.stt.events import STTTranscriptEvent

from vision_agents.core.llm.events import LLMResponseCompletedEvent

# from vision_agents.core.call.events import CallEndedEvent

# transcript_buffer = []



# question_count = 0
MAX_QUESTIONS = 5
# current_question = None
# current_answer_buffer = []
# conversation_segments = []
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

@app.get("/segments/{call_id}")
async def get_segments(call_id: str):
    session = sessions.get(call_id)

    if not session:
        return {"segments": []}

    return {
        "segments": session.get("segments", [])
    }

load_dotenv()

eval_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

async def create_agent(role: str):
    
    insturctions= f"""
    You are a professional mock interviewer for a {role}.
    Ask clear concise interview questions.
    Ask questions only, not delve into explaining them yourself by giving much examples and all.
    Keep responses short and natural.
    """

    agent = agents.Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Interview Coach", id="agent"),
        instructions=insturctions,
        llm=openrouter.LLM(
            # model = "openai/gpt-4o-mini"
            model = "qwen/qwen3-235b-a22b-thinking-2507"
        ),
        stt=deepgram.STT(),
        tts=deepgram.TTS(),
        turn_detection=smart_turn.TurnDetection(),
        
    )

    await agent.turn_detection.warmup()
    
    return agent

# async def join_call(agent: agents.Agent, call_type: str, call_id: str):
#     await agent.create_user()
#     call = await agent.create_call(call_type, call_id)

#     async with agent.join(call):
#         await agent.simple_response(
#             "Hello and welcome to your mock interview. Please introduce yourself."
#         )

#         await agent.finish()

async def evaluate_segment(call_id: str, question: str, answer: str):
    try:
        if call_id not in sessions:
            return None

        prompt = f"""
        You are evaluating a mock interview answer give technical and behavioural analysis.

        Question:
        {question}

        Answer:
        {answer}

        Return STRICT JSON:

        {{
        "short_feedback": "1 sentence feedback",
        "score": number 0-10
        }}
        """

        response = await eval_client.chat.completions.create(
            model="qwen/qwen3-235b-a22b-thinking-2507",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        raw = response.choices[0].message.content.strip()
        print("\nEVALUATION RAW:", raw)

        data = json.loads(raw)

        feedback = {
            "short_feedback": data.get("short_feedback", ""),
            "score": data.get("score", 0),
        }

        # ✅ Store feedback in session
        sessions[call_id]["latest_feedback"] = feedback

        print(f"\n[LOG] Feedback generated for call {call_id}:")
        print(json.dumps(feedback, indent=2))

        return feedback

    except Exception as e:
        print("Evaluation Error:", e)
        return None

async def join_call(agent: agents.Agent, call_type: str, call_id: str):
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)

    session = sessions[call_id]

    

    @agent.events.subscribe
    async def on_transcript(event: STTTranscriptEvent):
        session["current_answer_buffer"].append(event.text)

    @agent.events.subscribe
    async def on_turn_started(event: TurnStartedEvent):
        if event.participant and event.participant.user_id == "agent":
            return
        print(f"\n[LOG] User started speaking in call: {call_id}")

    @agent.events.subscribe
    async def on_turn_ended(event: TurnEndedEvent):
        if event.participant and event.participant.user_id == "agent":
            return
        
        print(f"\n[LOG] User finished speaking in call: {call_id}")

        if session["question_count"] >= 5:
            return

        full_answer = " ".join(session["current_answer_buffer"]).strip()
        session["current_answer_buffer"] = []

        if not full_answer:
            return

        question = session["current_question"]

        segment = {
            "question": question,
            "answer": full_answer,
        }

        session["segments"].append(segment)

        print(f"\n[LOG] Current segments for call {call_id}:")
        print(json.dumps(session["segments"], indent=2))

        asyncio.create_task(
            evaluate_segment(call_id, question, full_answer)
        )

        session["question_count"] += 1
        print(f"[LOG] Question count: {session['question_count']}/5")

    @agent.events.subscribe
    async def on_llm_response(event: LLMResponseCompletedEvent):
        session["current_question"] = event.text
        print(f"\n[LOG] Agent response (new question) for call {call_id}: {event.text}")

    try:
        async with agent.join(call):
            print("Agent joined call")

            session["current_question"] = "Please introduce yourself."

            await agent.say(
                "Hey! Welcome to your mock interview. Please introduce yourself."
            )

            while True:
                await asyncio.sleep(1)

    finally:
        active_agents.pop(call_id, None)
        print("Agent cleanup complete:", call_id)  
        
# if __name__ ==  "__main__":
#     role = "React Developer"
#     call_type = "default"
#     call_id = "test-call-123"

#     async def main():
#         agent = await create_agent(role)
#         await join_call(agent, call_type, call_id)
    
#     asyncio.run(main())

async def main_agent(role, call_type, call_id):
    agent = await create_agent(role)
    await join_call(agent, call_type, call_id)


# @app.on_event("startup")
# async def startup_event():
#     role = "React Developer"
#     call_type = "default"
#     call_id = "test-call-123"

#     asyncio.create_task(
#         main_agent(role, call_type, call_id)
#     )

# @app.post("/start-agent")
# async def start_agent():
#     role = "React Developer"
#     call_type = "default"
#     call_id = "test-call-123"

#     asyncio.create_task(
#         main_agent(role, call_type, call_id)
#     )

#     return {"status": "agent started"}

@app.post("/start-agent")
async def start_agent(data: dict):
    role = data.get("role", "React Developer")
    call_id = data.get("call_id")

    if not call_id:
        return {"error": "call_id required"}

    if call_id not in sessions:
        return {"error": "session not initialized"}

    if call_id in active_agents:
        return {"status": "already running"}

    active_agents[call_id] = "starting"

    async def runner():
        try:
            await main_agent(role, "default", call_id)
        finally:
            active_agents.pop(call_id, None)

    task = asyncio.create_task(runner())
    active_agents[call_id] = task

    return {"status": "agent started"}
    
@app.get("/latest-feedback/{call_id}")
async def get_latest_feedback(call_id: str):
    session = sessions.get(call_id)

    if not session:
        return {"feedback": None}

    return {
        "feedback": session.get("latest_feedback")
    }

@app.post("/create-session")
async def create_session():
    call_id = str(uuid.uuid4())

    sessions[call_id] = {
        "latest_feedback": None,
        "segments": [],
        "question_count": 0,
        "current_question": None,
        "current_answer_buffer": []
    }

    return {"call_id": call_id}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/ready")
async def ready():
    # If there's any warmup logic, it should go here.
    # For now, it's always ready if the server is up.
    return {"status": "ready"}