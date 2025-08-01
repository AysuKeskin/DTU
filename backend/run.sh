#!/bin/bash

# Python sanal ortamı oluştur
python3 -m venv venv

# Sanal ortamı aktive et
source venv/bin/activate

# Gereksinimleri yükle
pip install -r requirements.txt

# FastAPI uygulamasını debug modunda başlat
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug 