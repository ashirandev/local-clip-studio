FROM python:3.11-slim@sha256:b7a9e5f6a93d6d4a31f129bbd4bdee69efccb6ded0fb230d78aafaf37af98377
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg build-essential cmake libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "omnilingual-asr==0.2.0" "torch==2.8.0" "torchaudio==2.8.0"
RUN pip install --no-cache-dir "pyarrow==20.0.0"
RUN pip install --no-cache-dir "onnx==1.23.2" "onnxruntime==1.31.1"
WORKDIR /hearing
COPY export_onnx.py ./
COPY cards ./cards
ENV FAIRSEQ2_USER_ASSET_DIR=/hearing/cards PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 LCS_HEARING_CARD=omniASR_CTC_1B_v2@user
CMD ["python", "export_onnx.py"]
