import os
from dotenv import load_dotenv
from fastapi import FastAPI
from getstream import Stream

load_dotenv()

app = FastAPI()

STREAM_API_KEY = os.getenv("STREAM_API_KEY")
STREAM_API_SECRET = os.getenv("STREAM_API_SECRET")

stream_client = Stream(
    api_key=STREAM_API_KEY,
    api_secret=STREAM_API_SECRET,
)


@app.post("/create-token")
async def create_token(user_id: str):
    token = stream_client.create_token(user_id)
    return {
        "apiKey": STREAM_API_KEY,
        "token": token,
        "userId": user_id
    }