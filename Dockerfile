# Base leve com suporte a Python 3.10+
FROM python:3.10-slim

WORKDIR /app

# Dependências do sistema para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Instalação das libs Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código fonte
COPY src/ ./src/
COPY config/ ./config/

# Comando padrão (pode ser sobrescrito)
CMD ["python", "src/counter.py"]