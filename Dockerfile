FROM python:3.12-slim

# Chromium prints the PDF reports; Noto fonts render Arabic even without internet access.
RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium fonts-noto-core fonts-noto-color-emoji ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH CHROME_PATH=/usr/bin/chromium \
    KASHAF_HOSTED=1 PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
