FROM python:3.12

WORKDIR /data_visualiser

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0


COPY ./requirements.txt /data_visualiser/requirements.txt 
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY main.py /data_visualiser/main.py

CMD [ "python", "main.py" ]