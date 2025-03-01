import os
import subprocess
from base64 import b64encode
from threading import Semaphore
from typing import Any

import librosa
import numpy as np
import scipy
from flask import Flask, Response

app = Flask(__name__)

max_concurrent_requests = 5
sem = Semaphore(max_concurrent_requests)

STATUS_503 = Response({"error": "Too many ongoing jobs."}, status=503)

# constants for key detection
MAJOR_COEFFECIENTS = scipy.linalg.circulant(
    scipy.stats.zscore(
        np.asarray(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        )
    )
)

MINOR_COEFFECIENTS = scipy.linalg.circulant(
    scipy.stats.zscore(
        np.asarray(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        )
    )
)

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def feature_detection(id: str) -> tuple[Any, str]:
    # key detection
    y, sr = librosa.load(".data/" + id + ".wav")
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    major_correlation = MAJOR_COEFFECIENTS.T.dot(
        scipy.stats.zscore(chromagram.mean(axis=1))
    )
    minor_correlation = MINOR_COEFFECIENTS.T.dot(
        scipy.stats.zscore(chromagram.mean(axis=1))
    )

    best_major = KEYS[np.argmax(major_correlation)]
    best_minor = KEYS[np.argmax(minor_correlation)]

    if np.max(major_correlation) > np.max(minor_correlation):
        key = best_major + " major"
    else:
        key = best_minor + " minor"

    # tempo detection
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, sparse=True)
    if isinstance(tempo, np.ndarray):
        return tempo[0], key
    return tempo, key


@app.route("/by-search-term/<string:song_search_term>")
def by_search_term(song_search_term: str):
    if not sem.acquire(blocking=False):
        return STATUS_503

    try:
        id = b64encode(song_search_term.encode()).decode()
        # execute yt-dlp from subprocess to use the config file
        subprocess.run(
            ["yt-dlp", f"ytsearch:{song_search_term}", "-o", f".data/{id}.%(ext)s"]
        )
        tempo, key = feature_detection(id)

        os.remove(f".data/{id}.wav")

        return {"tempo": tempo, "key": key}
    finally:
        sem.release()


@app.route("/by-url/<string:song_url>")
def by_url(song_url: str):
    if not sem.acquire(blocking=False):
        return STATUS_503
    try:
        return "Hello, World!"
    finally:
        sem.release()

    return "Hello, World!"
