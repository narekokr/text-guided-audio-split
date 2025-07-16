# Use an official base image that supports Python 3.12.9
FROM python:3.12.9

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app
ARG OPENAPI_TOKEN
ENV API_KEY=${OPENAPI_TOKEN}
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install system dependencies (optional but recommended for building certain wheels)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
RUN pip install --no-cache-dir -r requirements.txt

RUN python3 -m pip install diffq
COPY . .
RUN pip3 install hf_xet python-multipart
RUN python3 test.py
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
