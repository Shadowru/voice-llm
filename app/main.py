import os
import json
import asyncio
import concurrent.futures
import tempfile
import io
import scipy.io.wavfile
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel
from llama_cpp import Llama

# --- Файл конфигурации ---
CONFIG_FILE = "/app/config.json"

# Дефолтные настройки
DEFAULT_CONFIG = {
    "system_prompt": "Ты дружелюбный помощник на мероприятии. Отвечай весело и коротко.",
    "voice_speaker": "xenia", # aidar, baya, kseniya, xenia, eugene
    "vad_threshold": 0.02,
    "silence_duration": 1500,
    "background_url": "/static/bg.jpg",
    "title_text": "Поговори с ИИ"
}

# Загрузка конфига
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return DEFAULT_CONFIG

def save_config(new_config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(new_config, f, ensure_ascii=False, indent=4)

# Глобальная переменная конфига (обновляется на лету)
current_config = load_config()

# --- Инициализация Моделей (как раньше) ---
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.gguf")
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "base")
SILERO_LOCAL_PATH = "/app/models/silero_v5_ru.pt"

print("Loading Models...")
stt_model = WhisperModel(WHISPER_SIZE, device="auto", compute_type="int8")

if os.path.exists(MODEL_PATH):
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)
else:
    llm = None

device = torch.device('cpu')
if not os.path.exists(SILERO_LOCAL_PATH):
    torch.hub.download_url_to_file("https://models.silero.ai/models/tts/ru/v5_ru.pt", SILERO_LOCAL_PATH)
silero_model = torch.package.PackageImporter(SILERO_LOCAL_PATH).load_pickle("tts_models", "model")
silero_model.to(device)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# --- API для Админки ---

class ConfigModel(BaseModel):
    system_prompt: str
    voice_speaker: str
    vad_threshold: float
    silence_duration: int
    background_url: str
    title_text: str

@app.get("/api/config")
def get_config():
    return current_config

@app.post("/api/config")
def update_config(config: ConfigModel):
    global current_config
    current_config = config.dict()
    save_config(current_config)
    return {"status": "ok", "config": current_config}

# --- Логика Обработки ---

async def process_audio_task(websocket: WebSocket, audio_data: bytes):
    # Используем current_config вместо хардкода
    
    # 1. STT
    loop = asyncio.get_event_loop()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    segments, _ = await loop.run_in_executor(executor, lambda: stt_model.transcribe(tmp_path, beam_size=5))
    text = " ".join([s.text for s in list(segments)]).strip()
    if tmp_path: os.remove(tmp_path)
    if not text: return

    await websocket.send_text(f"[User]: {text}")

    # 2. LLM
    if llm:
        # Берем промпт из конфига!
        sys_prompt = current_config["system_prompt"]
        
        prompt = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        stream = await loop.run_in_executor(executor, lambda: llm(prompt, max_tokens=256, stop=["<|eot_id|>"], stream=True))
        
        buffer = ""
        for output in stream:
            if asyncio.current_task().cancelled(): return
            token = output['choices'][0]['text']
            buffer += token
            if token in ['.', '!', '?', '\n'] and len(buffer.strip()) > 2:
                await generate_tts(websocket, buffer, loop)
                buffer = ""
        if buffer.strip():
            await generate_tts(websocket, buffer, loop)

async def generate_tts(websocket: WebSocket, text: str, loop):
    await websocket.send_text(f"[AI]: {text}")
    speaker = current_config["voice_speaker"] # Берем голос из конфига
    
    def _tts():
        return silero_model.apply_tts(text=text, speaker=speaker, sample_rate=48000)

    try:
        audio_tensor = await loop.run_in_executor(executor, _tts)
        buff = io.BytesIO()
        scipy.io.wavfile.write(buff, 48000, audio_tensor.numpy())
        await websocket.send_bytes(buff.getvalue())
    except Exception as e:
        print(f"TTS Error: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_task = None
    try:
        while True:
            data = await websocket.receive_bytes()
            if current_task and not current_task.done(): current_task.cancel()
            current_task = asyncio.create_task(process_audio_task(websocket, data))
    except WebSocketDisconnect:
        if current_task: current_task.cancel()