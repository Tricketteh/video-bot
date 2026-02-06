FROM python:3.10-slim

# System deps for yt-dlp (ffmpeg for merging, ca-certificates for TLS)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies
RUN pip install --no-cache-dir python-telegram-bot>=20.0 yt-dlp

# Copy bot
COPY bot.py /app/bot.py

# Example env placeholders
ENV BOT_TOKEN=""
ENV MAX_CONCURRENT_DOWNLOADS="2"
ENV MAX_BOT_FILE_BYTES="52428800"

CMD ["python", "bot.py"]
