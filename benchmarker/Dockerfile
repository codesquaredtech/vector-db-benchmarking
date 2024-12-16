FROM python:3.12

WORKDIR /benchmarker

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0


COPY ./requirements.txt /benchmarker/requirements.txt 
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY main.py /vectorizer/main.py

CMD [ "python", "main.py" ]