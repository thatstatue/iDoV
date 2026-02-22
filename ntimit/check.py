from ntimit.encrypter import voices
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import os
import time
from scipy import signal

# =========================
# CONFIG
# =========================

SAMPLE_RATE = 16000
BLOCK_DURATION = 0.24
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)  # 3840
STEP_SIZE = 80  # 5ms shift
DETECTION_THRESHOLD = 0.80  # strong match only

VOICE_SIGNATURES = []
HEX_DATA = []
CHAR_DATA = []

# =========================
# TEMPLATE LOADING
# =========================

def load_voice_signatures():
    global VOICE_SIGNATURES
    VOICE_SIGNATURES = []

    for vf in voices:
        data, sr = sf.read(vf, dtype="float32")

        if data.ndim > 1:
            data = np.mean(data, axis=1)

        if sr != SAMPLE_RATE:
            data = signal.resample(data, int(len(data) * SAMPLE_RATE / sr))

        # force exact block size
        if len(data) > BLOCK_SIZE:
            start = (len(data) - BLOCK_SIZE) // 2
            data = data[start:start + BLOCK_SIZE]
        elif len(data) < BLOCK_SIZE:
            pad = BLOCK_SIZE - len(data)
            data = np.pad(data, (0, pad))

        # normalize for cosine similarity
        data -= np.mean(data)
        norm = np.linalg.norm(data) + 1e-10
        data = data / norm

        VOICE_SIGNATURES.append(data.astype(np.float32))

    print(f"Loaded {len(VOICE_SIGNATURES)} templates")


# =========================
# DETECTION ENGINE
# =========================

class SynchronizedDetector:

    def __init__(self):
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.lock = threading.Lock()
        self.running = True

    def add_audio(self, samples):
        with self.lock:
            self.audio_buffer = np.concatenate([self.audio_buffer, samples])

    def process(self):
        global HEX_DATA

        while self.running:
            with self.lock:
                if len(self.audio_buffer) < BLOCK_SIZE:
                    continue

                buffer = self.audio_buffer.copy()

            i = 0
            consumed_until = 0

            while i + BLOCK_SIZE <= len(buffer):
                window = buffer[i:i + BLOCK_SIZE]

                # normalize window
                window = window - np.mean(window)
                norm = np.linalg.norm(window) + 1e-10
                window = window / norm

                best_score = -1
                best_idx = None

                for idx, template in enumerate(VOICE_SIGNATURES):
                    score = np.dot(window, template)
                    if score > best_score:
                        best_score = score
                        best_idx = idx

                if best_score > DETECTION_THRESHOLD:
                    print(f"✔ DETECTED {best_idx}  score={best_score:.3f}")
                    HEX_DATA.append(best_idx)
                    i += BLOCK_SIZE
                    consumed_until = i
                else:
                    i += STEP_SIZE

            # remove processed part from buffer
            if consumed_until > 0:
                with self.lock:
                    self.audio_buffer = self.audio_buffer[consumed_until:]

            time.sleep(0.01)


# =========================
# AUDIO LISTENER
# =========================

class AudioListener:

    def __init__(self, detector):
        self.detector = detector

    def callback(self, indata, frames, time_info, status):
        if status:
            print(status)

        samples = indata.flatten().astype(np.float32)
        self.detector.add_audio(samples)

    def start(self):
        with sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            callback=self.callback,
            blocksize=1024  # small block; alignment irrelevant now
        ):
            print("🎧 Listening...")
            while True:
                sd.sleep(1000)


# =========================
# HEX DECODING
# =========================

def decode_hex_stream():
    global HEX_DATA, CHAR_DATA

    i = 0
    out = []

    while i + 1 < len(HEX_DATA):
        a = HEX_DATA[i]
        b = HEX_DATA[i + 1]

        if 0 <= a < 16 and 0 <= b < 16:
            hex_pair = f"{a:x}{b:x}"
            char = bytes.fromhex(hex_pair).decode("utf-8", errors="replace")
            out.append(char)
            i += 2
        else:
            i += 1

    if out:
        text = "".join(out)
        CHAR_DATA.append(text)
        print("DECRYPTED:", text)

    HEX_DATA = []


def dummy_listener():
    device = 'pulse'
    sr = SAMPLE_RATE
    duration = 1000 * 15
    record_file = "recorded/REC_DELETE_LATER.wav"

    play_thread = Thread(target=play_wav, args=(play_file, device, False))
    play_thread.start()
    record_from_virtual_speaker(record_file, device, sr, duration)
    play_thread.join()
# =========================
# MAIN
# =========================

def main():
    load_voice_signatures()

    detector = SynchronizedDetector()

    processing_thread = threading.Thread(target=detector.process)
    processing_thread.daemon = True
    processing_thread.start()

    listener = AudioListener(detector)
    listener.start()


if __name__ == "__main__":
    main()