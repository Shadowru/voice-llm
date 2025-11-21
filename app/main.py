import os
import asyncio
import tempfile
import concurrent.futures
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from llama_cpp import Llama
import edge_tts

# Конфиг
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.gguf")
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "base")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Инициализация (глобальная)
print("Loading models...")
stt_model = WhisperModel(WHISPER_SIZE, device="auto", compute_type="int8")

if os.path.exists(MODEL_PATH):
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)
else:
    llm = None

# ThreadPool для тяжелых задач, чтобы не блокировать WebSocket
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

async def process_audio_task(websocket: WebSocket, audio_data: bytes):
    """Эта функция будет запускаться как отменяемая задача"""
    tmp_path = None
    try:
        # 1. STT (в отдельном потоке, чтобы не фризить event loop)
        loop = asyncio.get_event_loop()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        # Запускаем Whisper в экзекьюторе
        segments, _ = await loop.run_in_executor(executor, lambda: stt_model.transcribe(tmp_path, beam_size=5))
        text = " ".join([s.text for s in list(segments)]).strip() # list() нужен чтобы прочитать генератор
        
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

        if not text: return

        await websocket.send_text(f"[User]: {text}")

        # 2. LLM + TTS
        if llm:
            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            # Генератор токенов (это синхронный код, его сложно прервать мгновенно внутри, 
            # но мы проверяем asyncio.CancelledError между чанками TTS)
            stream = await loop.run_in_executor(executor, lambda: llm(prompt, max_tokens=256, stop=["<|eot_id|>"], stream=True))
            
            buffer = ""
            for output in stream:
                # Проверка отмены задачи (если пользователь перебил)
                if asyncio.current_task().cancelled():
                    print("Task cancelled during LLM generation")
                    return

                token = output['choices'][0]['text']
                buffer += token
                
                if token in ['.', '!', '?', '\n'] and len(buffer.strip()) > 2:
                    await send_audio(websocket, buffer)
                    buffer = ""
            
            if buffer.strip():
                await send_audio(websocket, buffer)
        
    except asyncio.CancelledError:
        print("Processing cancelled (User interrupted)")
        # Чистим мусор если отменили
        if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)
        raise # Важно пробросить, чтобы task корректно завершился
    except Exception as e:
        print(f"Error: {e}")
        if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)

async def send_audio(websocket: WebSocket, text: str):
    await websocket.send_text(f"[AI]: {text}")
    communicate = edge_tts.Communicate(text, "ru-RU-SvetlanaNeural")
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    try:
        await websocket.send_bytes(audio_data)
    except:
        pass # Сокет мог закрыться

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_task = None

    try:
        while True:
            # Ждем аудио от клиента
            data = await websocket.receive_bytes()
            
            # ЕСЛИ есть активная задача (бот говорит) -> ОТМЕНЯЕМ ЕЁ
            if current_task and not current_task.done():
                current_task.cancel()
                try:
                    await current_task 
                except asyncio.CancelledError:
                    pass # Это нормально
                print("Previous task cancelled.")

            # Запускаем новую задачу
            current_task = asyncio.create_task(process_audio_task(websocket, data))

    except WebSocketDisconnect:
        if current_task: current_task.cancel()
        print("Client disconnected")