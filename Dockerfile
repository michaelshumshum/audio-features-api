FROM python:3.12-slim

RUN apt-get update && apt-get install ffmpeg libsndfile1 -y --force-yes && \
    mkdir /app
WORKDIR /app

COPY main.py ./main.py
COPY requirements.txt ./requirements.txt
COPY yt-dlp.conf /etc/yt-dlp.conf

RUN mkdir .data &&\
    pip install -r requirements.txt

EXPOSE 8080
CMD ["flask", "--app", "main", "run", "--host=0.0.0.0", "--port=8080"]
