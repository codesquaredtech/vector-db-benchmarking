FROM nvidia/cuda:12.6.0-base-ubuntu22.04

WORKDIR /vectorizer

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /vectorizer/requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

COPY main.py /vectorizer/main.py

CMD ["python3", "main.py"]