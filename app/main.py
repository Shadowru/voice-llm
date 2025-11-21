import os
import asyncio
import tempfile
import concurrent.futures
import io
import scipy.io.wavfile
import torch # Добавили torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from llama_cpp import Llama

# --- КОНФИГУРАЦИЯ ---
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.gguf")
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "base")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Ты голосовой помощник.")

# Настройки Silero
SILERO_MODEL_URL = "https://models.silero.ai/models/tts/ru/v4_ru.pt"
SILERO_LOCAL_PATH = "/app/models/silero_v4_ru.pt"
SAMPLE_RATE = 48000
SPEAKER = "xenia" # Варианты: aidar, baya, kseniya, xenia, eugene, random

app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Глобальный экзекьютор
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

print("--- INIT ---")

# 1. Загрузка Whisper
print("Loading Whisper...")
stt_model = WhisperModel(WHISPER_SIZE, device="auto", compute_type="int8")

# 2. Загрузка LLM
print("Loading LLM...")
if os.path.exists(MODEL_PATH):
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)
else:
    llm = None
    print("WARNING: LLM model not found.")

# 3. Загрузка Silero TTS
print("Loading Silero TTS...")
if not os.path.exists(SILERO_LOCAL_PATH):
    print(f"Downloading Silero model to {SILERO_LOCAL_PATH}...")
    torch.hub.download_url_to_file(SILERO_MODEL_URL, SILERO_LOCAL_PATH)

device = torch.device('cpu') # Silero летает и на CPU
silero_model = torch.package.PackageImporter(SILERO_LOCAL_PATH).load_pickle("tts_models", "model")
silero_model.to(device)
print("Silero Ready.")

async def process_audio_task(websocket: WebSocket, audio_data: bytes):
    tmp_path = None
    try:
        loop = asyncio.get_event_loop()
        
        # --- STT ---
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        segments, _ = await loop.run_in_executor(executor, lambda: stt_model.transcribe(tmp_path, beam_size=5))
        text = " ".join([s.text for s in list(segments)]).strip()
        
        if tmp_path: os.remove(tmp_path)
        if not text: return

        await websocket.send_text(f"[User]: {text}")

        # --- LLM ---
        if llm:
            prompt = (
                f"<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            
            stream = await loop.run_in_executor(executor, lambda: llm(prompt, max_tokens=256, stop=["<|eot_id|>"], stream=True))
            
            buffer = ""
            for output in stream:
                if asyncio.current_task().cancelled(): return

                token = output['choices'][0]['text']
                buffer += token
                
                # Разбиваем на предложения для TTS
                if token in ['.', '!', '?', '\n'] and len(buffer.strip()) > 2:
                    await generate_and_send_silero(websocket, buffer, loop)
                    buffer = ""
            
            if buffer.strip():
                await generate_and_send_silero(websocket, buffer, loop)

    except asyncio.CancelledError:
        print("Task Cancelled")
        if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)
    except Exception as e:
        print(f"Error: {e}")

async def generate_and_send_silero(websocket: WebSocket, text: str, loop):
    """Генерация аудио через Silero и отправка WAV байтов"""
    await websocket.send_text(f"[AI]: {text}")
    
    def _tts():
        # Silero блокирует поток, поэтому запускаем внутри executor
        # apply_tts возвращает Tensor
        audio_tensor = silero_model.apply_tts(text=text,
                                              speaker=SPEAKER,
                                              sample_rate=SAMPLE_RATE)
        return audio_tensor

    try:
        # Запускаем в отдельном потоке
        audio_tensor = await loop.run_in_executor(executor, _tts)
        
        # Конвертируем Tensor -> WAV Bytes
        # Silero возвращает float32, браузеры это понимают
        buff = io.BytesIO()
        # Переводим в numpy и сохраняем как WAV
        scipy.io.wavfile.write(buff, SAMPLE_RATE, audio_tensor.numpy())
        
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
            if current_task and not current_task.done():
                current_task.cancel()
            current_task = asyncio.create_task(process_audio_task(websocket, data))
    except WebSocketDisconnect:
        if current_task: current_task.cancel()