FROM python:3.10-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Create data dirs
RUN mkdir -p uploads library attachments

EXPOSE 5000

CMD ["python", "app.py"]
