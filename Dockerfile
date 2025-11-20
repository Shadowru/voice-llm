# Используем образ с Python. Для GPU лучше взять nvidia/cuda base, 
# но для простоты берем python-slim (torch сам подтянет cuda libs если нужно)
FROM python:3.10-slim

# Установка системных зависимостей (нужны для сборки некоторых пакетов)
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/requirements.txt .
# Установка Python библиотек
# edge-tts используется вместо pyttsx3, так как он лучше работает в Docker (без X11)
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

# Порт для FastAPI
EXPOSE 8000

# Запуск
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]