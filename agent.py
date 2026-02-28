import os
import asyncio
import json
from dotenv import load_dotenv

from openai import AsyncOpenAI
from vision_agents.core import agents
from vision_agents.core.edge.types import User
from vision_agents.core.runner import Runner
from vision_agents.plugins import getstream, deepgram, cartesia, openrouter, smart_turn
from vision_agents.core.turn_detection.events import TurnStartedEvent, TurnEndedEvent
from vision_agents.core.stt.events import STTTranscriptEvent

from vision_agents.core.llm.events import LLMResponseCompletedEvent

# transcript_buffer = []
question_count = 0
MAX_QUESTIONS = 5
current_question = None
current_answer_buffer = []
conversation_segments = []

load_dotenv()

eval_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

async def create_agent(role: str):
    
    insturctions= f"""
    You are a professional mock interviewer for a {role}.
    Ask clear concise interview questions.
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
        tts=cartesia.TTS(),
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

async def evaluate_segment(question: str, answer: str):
    try:
        prompt = f"""
        You are evaluating a mock interview answer.

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

        short_feedback = data.get("short_feedback", "")
        score = data.get("score", 0)

        print("\n=== EVALUATION RESULT ===")
        print("Score:", score)
        print("Feedback:", short_feedback)
        print("=========================\n")

        return {
            "short_feedback": short_feedback,
            "score": score
        }

    except Exception as e:
        print("Evaluation Error:", e)
        return None

async def join_call(agent: agents.Agent, call_type: str, call_id: str):
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)
    

    @agent.events.subscribe
    async def on_turn_started(event: TurnStartedEvent):
        print(f"Speaker started talking (confidence: {event.confidence})")
    
    @agent.events.subscribe
    async def on_transcript(event: STTTranscriptEvent):
        global current_answer_buffer
        
        # Only capture final transcript
        current_answer_buffer.append(event.text)
        print("FINAL TRANSCRIPT:", event.text)


    @agent.events.subscribe
    async def on_turn_ended(event: TurnEndedEvent):
        global current_question
        global current_answer_buffer
        global conversation_segments
        global question_count

        print(f"Speaker finished (duration: {event.duration_ms}ms)")

        if event.participant and event.participant.user_id == "agent":
            print("Question:", current_question)
            return

        if question_count >= MAX_QUESTIONS:
            return

        full_answer = " ".join(current_answer_buffer).strip()
        current_answer_buffer = []

        if not full_answer:
            return

        print("\n====== SEGMENT ======")
        print("Question:", current_question)
        print("Answer:", full_answer)
        print("=====================\n")

        # Save segment
        segment = {
            "question": current_question,
            "answer": full_answer,
        }

        conversation_segments.append(segment)

        print("Saved segment:", segment)

        # Update state
        asyncio.create_task(
            evaluate_segment(current_question, full_answer)
        )

        question_count += 1

    @agent.events.subscribe
    async def on_llm_response(event: LLMResponseCompletedEvent):
        global current_question

        # The agent's spoken response becomes the new question
        current_question = event.text
        print("NEW QUESTION SET:", current_question)

    async with agent.join(call):
        print("Agent joined call")

        current_question = "Please introduce yourself."

        await agent.say(
            "Hey, hello ! I hope you are fine. Welcome to your mock interview. Please introduce yourself."
        )

        while True:
            await asyncio.sleep(1)
    
    
        
if __name__ ==  "__main__":
    role = "React Developer"
    call_type = "default"
    call_id = "test-call-123"

    async def main():
        agent = await create_agent(role)
        await join_call(agent, call_type, call_id)
    
    asyncio.run(main())