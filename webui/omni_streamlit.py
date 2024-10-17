import streamlit as st
import wave

# from ASR import recognize
import requests
import pyaudio
import numpy as np
import base64
import io
import os
import time
import tempfile
import librosa
import traceback
from pydub import AudioSegment
import sys

# Get the root directory of the project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the project root directory to sys.path
sys.path.append(project_root)

from utils.vad import get_speech_timestamps, collect_chunks, VadOptions


API_URL = os.getenv("API_URL", "http://127.0.0.1:60808/chat")

# recording parameters
IN_FORMAT = pyaudio.paInt16
IN_CHANNELS = 1
IN_RATE = 24000
IN_CHUNK = 1024
IN_SAMPLE_WIDTH = 2
VAD_STRIDE = 0.5

# playing parameters
OUT_FORMAT = pyaudio.paInt16
OUT_CHANNELS = 1
OUT_RATE = 24000
OUT_SAMPLE_WIDTH = 2
OUT_CHUNK = 5760


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


def run_vad(ori_audio, sr):
    _st = time.time()
    try:
        audio = np.frombuffer(ori_audio, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
        sampling_rate = 16000
        if sr != sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)

        vad_parameters = {}
        vad_parameters = VadOptions(**vad_parameters)
        speech_chunks = get_speech_timestamps(audio, vad_parameters)
        audio = collect_chunks(audio, speech_chunks)
        duration_after_vad = audio.shape[0] / sampling_rate

        if sr != sampling_rate:
            # resample to original sampling rate
            vad_audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=sr)
        else:
            vad_audio = audio
        vad_audio = np.round(vad_audio * 32768.0).astype(np.int16)
        vad_audio_bytes = vad_audio.tobytes()

        return duration_after_vad, vad_audio_bytes, round(time.time() - _st, 4)
    except Exception as e:
        msg = f"[asr vad error] audio_len: {len(ori_audio)/(sr*2):.3f} s, trace: {traceback.format_exc()}"
        print(msg)
        return -1, ori_audio, round(time.time() - _st, 4)


def warm_up():
    frames = b"\x00\x00" * 1024 * 2  # 1024 frames of 2 bytes each
    dur, frames, tcost = run_vad(frames, 16000)
    print(f"warm up done, time_cost: {tcost:.3f} s")


def save_tmp_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        file_name = tmpfile.name
        audio = AudioSegment(
            data=audio_bytes,
            sample_width=OUT_SAMPLE_WIDTH,
            frame_rate=OUT_RATE,
            channels=OUT_CHANNELS,
        )
        audio.export(file_name, format="wav")
        return file_name


def speaking(status):

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open PyAudio stream
    stream = p.open(
        format=OUT_FORMAT, channels=OUT_CHANNELS, rate=OUT_RATE, output=True
    )

    audio_buffer = io.BytesIO()
    wf = wave.open(audio_buffer, "wb")
    wf.setnchannels(IN_CHANNELS)
    wf.setsampwidth(IN_SAMPLE_WIDTH)
    wf.setframerate(IN_RATE)
    total_frames = b"".join(st.session_state.frames)
    dur = len(total_frames) / (IN_RATE * IN_CHANNELS * IN_SAMPLE_WIDTH)
    status.warning(f"Speaking... recorded audio duration: {dur:.3f} s")
    wf.writeframes(total_frames)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        with open(tmpfile.name, "wb") as f:
            f.write(audio_buffer.getvalue())
        file_name = tmpfile.name
        with st.chat_message("user"):
            st.audio(file_name, format="audio/wav", loop=False, autoplay=False)
        st.session_state.messages.append(
            {"role": "assistant", "content": file_name, "type": "audio"}
        )

    st.session_state.frames = []

    audio_bytes = audio_buffer.getvalue()
    base64_encoded = str(base64.b64encode(audio_bytes), encoding="utf-8")
    files = {"audio": base64_encoded}
    output_audio_bytes = b""
    with requests.post(API_URL, json=files, stream=True) as response:
        try:
            for chunk in response.iter_content(chunk_size=OUT_CHUNK):
                if chunk:
                    # Convert chunk to numpy array
                    output_audio_bytes += chunk
                    audio_data = np.frombuffer(chunk, dtype=np.int8)
                    # Play audio
                    stream.write(audio_data)
        except Exception as e:
            st.error(f"Error during audio streaming: {e}")

    out_file = save_tmp_audio(output_audio_bytes)
    with st.chat_message("assistant"):
        st.audio(out_file, format="audio/wav", loop=False, autoplay=False)
    st.session_state.messages.append(
        {"role": "assistant", "content": out_file, "type": "audio"}
    )

    wf.close()
    # Close PyAudio stream and terminate PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
    st.session_state.speaking = False
    st.session_state.recording = True


def recording(status):
    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=IN_FORMAT,
        channels=IN_CHANNELS,
        rate=IN_RATE,
        input=True,
        frames_per_buffer=IN_CHUNK,
    )

    temp_audio = b""
    vad_audio = b""

    start_talking = False
    last_temp_audio = None
    st.session_state.frames = []

    while st.session_state.recording:
        status.success("Listening...")
        audio_bytes = stream.read(IN_CHUNK)
        temp_audio += audio_bytes

        if len(temp_audio) > IN_SAMPLE_WIDTH * IN_RATE * IN_CHANNELS * VAD_STRIDE:
            dur_vad, vad_audio_bytes, time_vad = run_vad(temp_audio, IN_RATE)

            print(f"duration_after_vad: {dur_vad:.3f} s, time_vad: {time_vad:.3f} s")

            if dur_vad > 0.2 and not start_talking:
                if last_temp_audio is not None:
                    st.session_state.frames.append(last_temp_audio)
                start_talking = True
            if start_talking:
                st.session_state.frames.append(temp_audio)
            if dur_vad < 0.1 and start_talking:
                st.session_state.recording = False
                print(f"speech end detected. excit")
            last_temp_audio = temp_audio
            temp_audio = b""

    stream.stop_stream()
    stream.close()

    audio.terminate()


def main():

    st.title("Chat Mini-Omni Demo")
    status = st.empty()

    if "warm_up" not in st.session_state:
        warm_up()
        st.session_state.warm_up = True
    if "start" not in st.session_state:
        st.session_state.start = False
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "speaking" not in st.session_state:
        st.session_state.speaking = False
    if "frames" not in st.session_state:
        st.session_state.frames = []

    if not st.session_state.start:
        status.warning("Click Start to chat")

    start_col, stop_col, _ = st.columns([0.2, 0.2, 0.6])
    start_button = start_col.button("Start", key="start_button")
    # stop_button = stop_col.button("Stop", key="stop_button")
    if start_button:
        time.sleep(1)
        st.session_state.recording = True
        st.session_state.start = True

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["type"] == "msg":
                st.markdown(message["content"])
            elif message["type"] == "img":
                st.image(message["content"], width=300)
            elif message["type"] == "audio":
                st.audio(
                    message["content"], format="audio/wav", loop=False, autoplay=False
                )

    while st.session_state.start:
        if st.session_state.recording:
            recording(status)

        if not st.session_state.recording and st.session_state.start:
            st.session_state.speaking = True
            speaking(status)

        # if stop_button:
        #     status.warning("Stopped, click Start to chat")
        #     st.session_state.start = False
        #     st.session_state.recording = False
        #     st.session_state.frames = []
        #     break


if __name__ == "__main__":
    main()
