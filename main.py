from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from openai import OpenAI
import os
import hashlib
import asyncio
from datetime import datetime
import uvicorn

# Load env variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize FastAPI app
app = FastAPI(title="AI Interview Platform", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for audio
os.makedirs("static/audio", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Request/Response Models
class ChatMessage(BaseModel):
    message: str
    round_type: str
    history: List[Dict[str, Any]] = []

class ChatResponse(BaseModel):
    response: str
    audio_url: Optional[str] = None

# Interview prompts
INTERVIEW_PROMPTS = {
    "technical": """You are an experienced technical interviewer conducting a coding interview...
Current conversation context: This is a technical interview round.""",
    "hr": """You are an experienced HR interviewer conducting a behavioral interview...
Current conversation context: This is an HR/behavioral interview round.""",
    "system-design": """You are a senior software architect conducting a system design interview...
Current conversation context: This is a system design interview round.""",
    "case-study": """You are a business consultant conducting a case study interview...
Current conversation context: This is a case study/business problem-solving interview round."""
}

def get_system_prompt(round_type: str) -> str:
    return INTERVIEW_PROMPTS.get(round_type, INTERVIEW_PROMPTS["technical"])

async def generate_response(message: str, round_type: str, history: List[Dict]) -> str:
    """Generate AI response using OpenAI GPT"""
    try:
        messages = [
            {"role": "system", "content": get_system_prompt(round_type)}
        ]

        # Keep only last 10 messages
        for msg in history[-10:]:
            if msg["type"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["type"] == "bot":
                messages.append({"role": "assistant", "content": msg["content"]})

        messages.append({"role": "user", "content": message})

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error generating response: {e}")
        return "I apologize, but I'm having trouble generating a response right now."

async def generate_audio(text: str) -> str:
    """Generate audio using OpenAI TTS"""
    try:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        audio_filename = f"audio_{text_hash}.mp3"
        audio_path = f"static/audio/{audio_filename}"

        if os.path.exists(audio_path):
            return f"/static/audio/{audio_filename}"

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.audio.speech.create(
                model="gpt-4o-mini-tts",  # or "tts-1"
                voice="alloy",
                input=text
            )
        )

        with open(audio_path, "wb") as f:
            f.write(response.read())

        return f"/static/audio/{audio_filename}"

    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """Main chat endpoint for interview interactions"""
    try:
        ai_response = await generate_response(
            chat_message.message,
            chat_message.round_type,
            chat_message.history
        )

        audio_url = await generate_audio(ai_response)

        return ChatResponse(
            response=ai_response,
            audio_url=audio_url
        )

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/")
async def root():
    return {"message": "AI Interview Platform API", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
