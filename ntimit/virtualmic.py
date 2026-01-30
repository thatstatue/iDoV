from threading import Thread

import sounddevice as sd
import webrtcvad
import soundfile as sf
import numpy as np
LENGTH = (12+2)*20
def print_devices():
    for i, d in enumerate(sd.query_devices()):
        print(i, d)

def play_wav(wav_path, device=None):
    data, sr = sf.read(wav_path, dtype='float32')
    sd.play(data, sr, device=device)
    #sd.wait()


def play_and_record_with_vad(wav_path, output_recorded, vad_mode=2, frame_ms=1):
    vad = webrtcvad.Vad(vad_mode)
    data, sr = sf.read(wav_path, dtype='int16')
    if data.ndim > 1:
        data = data[:, 0]
    samples_per_frame = int(sr * frame_ms / 1000)
    recorded_frames = []

    def callback(indata,  outdata, frames_count, time, status):
        outdata[:] = indata[:]
        frame_bytes = indata.tobytes()
        if vad.is_speech(frame_bytes, sr):
            recorded_frames.append(indata.copy())
    with sd.Stream(samplerate=sr, blocksize=samples_per_frame, dtype='int16', channels=1, callback=callback):
        sd.play(data, sr)
        sd.wait()

    if recorded_frames:
        print("yayayayyayayayayay")
        rec = np.concatenate(recorded_frames)
        sf.write(output_recorded, rec, sr)
    else:
        print("hohohohohooohohohoho")

def play_to_virtual_mic(wav_path, virtual_mic_device):
    data, sr = sf.read(wav_path, dtype='float32')
    if data.ndim > 1:
        data = data[:, 0]  # mono

    sd.play(data, sr, device=virtual_mic_device,blocking=True)
    sd.wait()


def record_from_virtual_speaker(output_file, device, samplerate=48000):
    frames = []
    def callback(indata, frames_count, time, status):
        frames.append(indata.copy())

    with sd.InputStream(
            device=device,
            channels=2,
            samplerate=48000,
            callback=callback):
        sd.sleep(LENGTH)
    audio = b''.join([f.tobytes() for f in frames])
    audio_np = np.concatenate(frames, axis=0)

    sf.write(output_file, audio_np, samplerate)


if __name__ == "__main__":
    MONITOR = 'pulse'
    Thread(play_wav("test/SA1-3_115.WAV", device='pulse')).start()
    Thread(record_from_virtual_speaker("recorded/REC_SA1-3_115.WAV", device='pulse')).start()
    #play_and_record_with_vad("recorded/REC_SA1-3_115.wav","recorded/REC_SA1-3_115_VAD.wav", 2, 20)


