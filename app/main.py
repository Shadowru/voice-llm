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
from pydantic import BaseModel
from faster_whisper import WhisperModel
from llama_cpp import Llama
from openai import AsyncOpenAI

# --- КОНФИГУРАЦИЯ ---
CONFIG_FILE = "/app/config.json"

# Дефолтные настройки (если файл пустой или сломан)
DEFAULT_CONFIG = {
    "system_prompt": "Ты голосовой помощник.",
    "voice_speaker": "xenia",
    "vad_threshold": 0.02,
    "silence_duration": 1500,
    "background_url": "",
    "title_text": "AI Assistant",
    "sample_rate": 48000  # Silero поддерживает: 8000, 24000, 48000
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Объединяем с дефолтным, чтобы не терять новые ключи
                return {**DEFAULT_CONFIG, **data}
        except Exception as e:
            print(f"Error loading config: {e}")
    return DEFAULT_CONFIG

def save_config(new_config):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(new_config, f, ensure_ascii=False, indent=4)
        print("Config saved successfully.")
    except Exception as e:
        print(f"Error saving config: {e}")

# Загружаем конфиг при старте
current_config = load_config()

# --- ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ (.env) ---
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.gguf")
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "base")
# Настройки Silero из .env
SILERO_LOCAL_PATH = os.getenv("SILERO_MODEL_PATH", "/app/models/silero_v4_ru.pt")
SILERO_MODEL_URL = os.getenv("SILERO_MODEL_URL", "https://models.silero.ai/models/tts/ru/v4_ru.pt")

# Новые переменные
LLM_CTX_SIZE = int(os.getenv("LLM_CTX_SIZE", "2048"))
USE_OPENAI_API = os.getenv("USE_OPENAI_API", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-xxx")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# --- ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ ---
print("--- INIT MODELS ---")
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
device = torch.device('cpu')

# 1. Whisper
print(f"Loading Whisper: {WHISPER_SIZE}...")
stt_model = WhisperModel(WHISPER_SIZE, device="auto", compute_type="int8")

# 2. LLM
llm_local = None
aclient = None

if USE_OPENAI_API:
    print(f"Connecting to OpenAI API: {OPENAI_BASE_URL} [{OPENAI_MODEL}]")
    aclient = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
else:
    print(f"Loading Local LLM: {MODEL_PATH} (Ctx: {LLM_CTX_SIZE})...")
    if os.path.exists(MODEL_PATH):
        llm_local = Llama(
            model_path=MODEL_PATH, 
            n_ctx=LLM_CTX_SIZE, # Используем переменную из .env
            n_gpu_layers=-1, 
            verbose=False
        )
    else:
        print("WARNING: Local model file not found!")

# 3. Silero TTS
print(f"Loading Silero from: {SILERO_LOCAL_PATH}")

# Проверяем наличие файла. Если нет - качаем по ссылке из ENV
if not os.path.exists(SILERO_LOCAL_PATH):
    print(f"Model not found. Downloading from {SILERO_MODEL_URL}...")
    try:
        torch.hub.download_url_to_file(SILERO_MODEL_URL, SILERO_LOCAL_PATH)
        print("Download complete.")
    except Exception as e:
        print(f"ERROR downloading Silero model: {e}")
        # Можно добавить выход, если модель критична
        exit(1)

if os.path.exists(SILERO_LOCAL_PATH):
    silero_model = torch.package.PackageImporter(SILERO_LOCAL_PATH).load_pickle("tts_models", "model")
    silero_model.to(device)
    print("Silero Ready.")
else:
    print("ERROR: Silero model file is missing and could not be downloaded.")
    silero_model = None

app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# --- API CONFIG ---

class ConfigModel(BaseModel):
    system_prompt: str
    voice_speaker: str
    vad_threshold: float
    silence_duration: int
    background_url: str
    title_text: str
    sample_rate: int # Добавили в валидацию

@app.get("/api/config")
def get_config_endpoint():
    return current_config

@app.post("/api/config")
def update_config_endpoint(config: ConfigModel):
    global current_config
    current_config = config.dict()
    save_config(current_config)
    return {"status": "ok", "config": current_config}

# --- ЛОГИКА ---

async def run_llm_pipeline(websocket: WebSocket, text: str):
    if not text: return
    await websocket.send_text(f"[User]: {text}")

    sys_prompt = current_config["system_prompt"]
    loop = asyncio.get_event_loop()
    
    # Генератор потока (зависит от выбранного бэкенда)
    stream_iterator = None

    if USE_OPENAI_API and aclient:
        # --- Ветка OpenAI API ---
        try:
            stream = await aclient.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": text}
                ],
                stream=True
            )
            # Адаптер для унификации
            async def openai_adapter():
                async for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content: yield content
            stream_iterator = openai_adapter()
            
        except Exception as e:
            print(f"API Error: {e}")
            await websocket.send_text(f"[Error]: API connection failed: {e}")
            return

    elif llm_local:
        # --- Ветка Local Llama ---
        prompt = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        # Запускаем синхронный генератор в executor'е, но читаем его асинхронно? 
        # Llama-cpp stream синхронный. Лучше использовать итератор.
        def local_gen():
            return llm_local(prompt, max_tokens=512, stop=["<|eot_id|>"], stream=True)
            
        stream_obj = await loop.run_in_executor(executor, local_gen)
        
        async def local_adapter():
            for output in stream_obj:
                # Проверка отмены задачи внутри синхронного цикла сложна, 
                # но мы проверяем asyncio.current_task() в цикле потребления
                yield output['choices'][0]['text']
        stream_iterator = local_adapter()
    
    else:
        await websocket.send_text("[Error]: No LLM loaded.")
        return

    # --- ОБЩИЙ ЦИКЛ ОБРАБОТКИ ТОКЕНОВ ---
    buffer = ""
    try:
        async for token in stream_iterator:
            if asyncio.current_task().cancelled(): 
                print("Generation cancelled.")
                return

            buffer += token
            
            # Эвристика конца предложения
            if token in ['.', '!', '?', '\n', ';'] and len(buffer.strip()) > 5:
                # Отправляем на озвучку
                await generate_tts(websocket, buffer, loop)
                buffer = ""
        
        # Остаток
        if buffer.strip():
            await generate_tts(websocket, buffer, loop)
            
    except Exception as e:
        print(f"Stream Error: {e}")

async def generate_tts(websocket: WebSocket, text: str, loop):
    await websocket.send_text(f"[AI]: {text}")
    
    # Берем настройки из конфига
    speaker = current_config["voice_speaker"]
    sample_rate = int(current_config.get("sample_rate", 48000))
    
    # Защита от дурака (Silero поддерживает только эти рейты)
    if sample_rate not in [8000, 24000, 48000]:
        sample_rate = 48000

    def _tts():
        return silero_model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate)

    try:
        audio_tensor = await loop.run_in_executor(executor, _tts)
        buff = io.BytesIO()
        # Используем тот же sample_rate для записи WAV
        scipy.io.wavfile.write(buff, sample_rate, audio_tensor.numpy())
        await websocket.send_bytes(buff.getvalue())
    except Exception as e:
        print(f"TTS Error: {e}")

# --- TASKS ---

async def process_audio_task(websocket: WebSocket, audio_data: bytes):
    loop = asyncio.get_event_loop()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    segments, _ = await loop.run_in_executor(executor, lambda: stt_model.transcribe(tmp_path, beam_size=5))
    text = " ".join([s.text for s in list(segments)]).strip()
    if tmp_path: os.remove(tmp_path)
    
    await run_llm_pipeline(websocket, text)

async def process_text_task(websocket: WebSocket, text_data: str):
    await run_llm_pipeline(websocket, text_data)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_task = None
    try:
        while True:
            message = await websocket.receive()
            if current_task and not current_task.done():
                current_task.cancel()

            if "bytes" in message and message["bytes"]:
                current_task = asyncio.create_task(process_audio_task(websocket, message["bytes"]))
            elif "text" in message and message["text"]:
                current_task = asyncio.create_task(process_text_task(websocket, message["text"]))

    except WebSocketDisconnect:
        if current_task: current_task.cancel()