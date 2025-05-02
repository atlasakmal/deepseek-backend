from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("NVAPI_KEY"),
    base_url="https://integrate.api.nvidia.com/v1"
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    def stream_response():
        completion = client.chat.completions.create(
            model="deepseek-ai/deepseek-r1",
            messages=messages,
            stream=True
        )
        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return StreamingResponse(stream_response(), media_type="text/plain")