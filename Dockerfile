FROM python:3.11-slim

WORKDIR /app

# System deps for Scapy (libpcap) and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make sure the data + model + static/img dirs exist
RUN mkdir -p data models static/img

# Train on startup if no models exist yet
CMD ["sh", "-c", \
     "python scripts/train_classical.py && \
      gunicorn --bind 0.0.0.0:5050 --workers 2 --timeout 120 app:app"]

EXPOSE 5050