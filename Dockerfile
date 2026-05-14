FROM python:3.10-slim

RUN pip install --no-cache-dir \
    numpy \
    opencv-python-headless \
    mediapipe \
    scipy \
    matplotlib \
    pandas

WORKDIR /workdir
