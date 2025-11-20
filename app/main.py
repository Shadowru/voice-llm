import os
import asyncio
import tempfile
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from llama_cpp import Llama
import edge_tts

# Читаем конфиг из ENV
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.gguf")
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "base")

app = FastAPI()

print(f"--- Config ---")
print(f"Model: {MODEL_PATH}")
print(f"Whisper: {WHISPER_SIZE}")

# Инициализация Whisper
print("Loading Whisper...")
stt_model = WhisperModel(WHISPER_SIZE, device="auto", compute_type="int8")

# Инициализация LLM
print("Loading LLM...")
if os.path.exists(MODEL_PATH):
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)
else:
    print(f"ERROR: Model not found at {MODEL_PATH}")
    llm = None

app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            
            # 1. Сохраняем аудио во временный файл
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            # 2. STT
            segments, _ = stt_model.transcribe(tmp_path, beam_size=5)
            text = " ".join([s.text for s in segments]).strip()
            os.remove(tmp_path)

            if not text: continue

            await websocket.send_text(f"[User]: {text}")

            # 3. LLM + TTS Stream
            if llm:
                prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                stream = llm(prompt, max_tokens=256, stop=["<|eot_id|>"], stream=True)
                
                buffer = ""
                for output in stream:
                    token = output['choices'][0]['text']
                    buffer += token
                    # Разбиваем по знакам препинания
                    if token in ['.', '!', '?', '\n'] and len(buffer.strip()) > 2:
                        await send_audio(websocket, buffer)
                        buffer = ""
                
                if buffer.strip():
                    await send_audio(websocket, buffer)
            else:
                await send_audio(websocket, f"Модель не загружена. Вы сказали: {text}")

    except WebSocketDisconnect:
        print("Client disconnected")

async def send_audio(websocket: WebSocket, text: str):
    await websocket.send_text(f"[AI]: {text}")
    communicate = edge_tts.Communicate(text, "ru-RU-SvetlanaNeural")
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    await websocket.send_bytes(audio_data)