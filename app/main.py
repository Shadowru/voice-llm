import os
import json
import asyncio
import concurrent.futures
import tempfile
import io
import scipy.io.wavfile
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel
from llama_cpp import Llama

# --- КОНФИГУРАЦИЯ ---
CONFIG_FILE = "/app/config.json"
DEFAULT_CONFIG = {
    "system_prompt": "Ты киберпанк-ассистент.",
    "voice_speaker": "xenia",
    "vad_threshold": 0.02,
    "silence_duration": 1500,
    "background_url": "",
    "title_text": "AI CORE"
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return DEFAULT_CONFIG

current_config = load_config()

# --- INIT MODELS ---
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.gguf")
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "base")
SILERO_LOCAL_PATH = "/app/models/silero_v4_ru.pt"

print("Loading Models...")
stt_model = WhisperModel(WHISPER_SIZE, device="auto", compute_type="int8")

if os.path.exists(MODEL_PATH):
    llm = Llama(model_path=MODEL_PATH, n_ctx=8192, n_gpu_layers=-1, verbose=False)
else:
    llm = None

device = torch.device('cpu')
if not os.path.exists(SILERO_LOCAL_PATH):
    torch.hub.download_url_to_file("https://models.silero.ai/models/tts/ru/v4_ru.pt", SILERO_LOCAL_PATH)
silero_model = torch.package.PackageImporter(SILERO_LOCAL_PATH).load_pickle("tts_models", "model")
silero_model.to(device)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# --- API Config ---
@app.get("/api/config")
def get_config(): return current_config

# --- ОБЩАЯ ФУНКЦИЯ ГЕНЕРАЦИИ (LLM + TTS) ---
async def run_llm_pipeline(websocket: WebSocket, text: str):
    """Принимает текст (от Whisper или из отладки) и генерирует ответ"""
    if not text: return
    
    # Отправляем клиенту, что мы поняли (для отображения в чате)
    await websocket.send_text(f"[User]: {text}")

    if not llm: return

    loop = asyncio.get_event_loop()
    sys_prompt = current_config["system_prompt"]
    
    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    # Генерация потока токенов
    stream = await loop.run_in_executor(executor, lambda: llm(prompt, max_tokens=256, stop=["<|eot_id|>"], stream=True))
    
    buffer = ""
    for output in stream:
        if asyncio.current_task().cancelled(): return
        token = output['choices'][0]['text']
        buffer += token
        
        # Разбиваем на предложения для TTS
        if token in ['.', '!', '?', '\n'] and len(buffer.strip()) > 2:
            await generate_tts(websocket, buffer, loop)
            buffer = ""
            
    if buffer.strip():
        await generate_tts(websocket, buffer, loop)

async def generate_tts(websocket: WebSocket, text: str, loop):
    await websocket.send_text(f"[AI]: {text}")
    speaker = current_config["voice_speaker"]
    
    def _tts():
        return silero_model.apply_tts(text=text, speaker=speaker, sample_rate=48000)

    try:
        audio_tensor = await loop.run_in_executor(executor, _tts)
        buff = io.BytesIO()
        scipy.io.wavfile.write(buff, 48000, audio_tensor.numpy())
        await websocket.send_bytes(buff.getvalue())
    except Exception as e:
        print(f"TTS Error: {e}")

# --- ОБРАБОТЧИКИ ЗАДАЧ ---

async def process_audio_task(websocket: WebSocket, audio_data: bytes):
    """1. Распознает аудио -> 2. Запускает пайплайн"""
    loop = asyncio.get_event_loop()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    segments, _ = await loop.run_in_executor(executor, lambda: stt_model.transcribe(tmp_path, beam_size=5))
    text = " ".join([s.text for s in list(segments)]).strip()
    
    if tmp_path: os.remove(tmp_path)
    
    # Передаем распознанный текст в LLM
    await run_llm_pipeline(websocket, text)

async def process_text_task(websocket: WebSocket, text_data: str):
    """Прямой запуск пайплайна (для отладки)"""
    await run_llm_pipeline(websocket, text_data)

# --- WEBSOCKET ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_task = None
    try:
        while True:
            # Используем receive() чтобы принимать И текст И байты
            message = await websocket.receive()
            
            if current_task and not current_task.done():
                current_task.cancel()

            if "bytes" in message and message["bytes"]:
                # Пришло аудио
                current_task = asyncio.create_task(process_audio_task(websocket, message["bytes"]))
            
            elif "text" in message and message["text"]:
                # Пришел текст (отладка)
                current_task = asyncio.create_task(process_text_task(websocket, message["text"]))

    except WebSocketDisconnect:
        if current_task: current_task.cancel()