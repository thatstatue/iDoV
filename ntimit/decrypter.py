from ntimit.consts import BLOCK_SIZE, BLOCK_SIZE_SECONDS, SILENCE_THRESHOLD, VOICE_SIGNATURES, SAMPLE_RATE, INPUT_DEVICE, OUTPUT_DIR
from ntimit.utilities import decrypt_voice, load_voice_signatures
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
import os


def hex_to_text(hex_str):
    try:
        v = bytes.fromhex(hex_str).decode('utf-8')

        return v
    except:
        return "?"

class AudioListener:
    def __init__(self, input_device=None, samplerate=16000, output_dir="recorded"):
        self.input_device = input_device
        self.samplerate = samplerate
        self.output_dir = output_dir
        self.block_duration = BLOCK_SIZE_SECONDS
        self.block_size = BLOCK_SIZE  # samples per block (3840)
        self._sample_buffer = np.zeros(0, dtype=np.float32)
        os.makedirs(output_dir, exist_ok=True)

        # Recording state
        self.listening = False
        self.recording_frames = []
        self.silent_blocks_count = 0
        self.silence_threshold = SILENCE_THRESHOLD
        self.energy_threshold = 0.001
        self.current_recording_id = 0
        self.running = True

        # Buffer to accumulate exact blocks for processing
        self._process_queue = []  # list of full-size numpy arrays

    def _is_silent(self, audio_chunk):
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        return rms < self.energy_threshold

    def _enqueue_block(self, block):
        if len(block) != self.block_size:
            # If block delivered shorter/longer, center-trim or pad
            if len(block) > self.block_size:
                start = (len(block) - self.block_size) // 2
                block = block[start:start + self.block_size]
            else:
                pad = self.block_size - len(block)
                left = pad // 2
                right = pad - left
                block = np.pad(block, (left, right), mode='constant')
        self._process_queue.append(block)


    def audio_callback(self, indata, frames, time_, status):
        if status: print("Audio callback status:", status)
        is_silent = self._is_silent(indata)
        if not is_silent:
            if not self.listening:
               x = 1# print("audio callback listening. please be patient...")

            self.listening = True
            self.silent_blocks_count = 0

            samples = indata.flatten().astype(np.float32)
            self._sample_buffer = np.concatenate((self._sample_buffer, samples))
            self.recording_frames.append(indata.copy())
        else:  # silent block

            if self.listening:
                self.silent_blocks_count += 1
                if self.silent_blocks_count >= self.silence_threshold:
                    print("silence detected")
                    self._stop_recording()
                    self.listening = False

    def _stop_recording(self):
        if self.recording_frames:
            audio_np = np.concatenate(self.recording_frames)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"{self.output_dir}/recording_{timestamp}_{self.current_recording_id}.wav"
            sf.write(filename, audio_np, self.samplerate)
            duration = len(audio_np) / self.samplerate
            print(f"✅ Recording saved: {filename} (duration: {duration:.2f}s)")
            decrypt_voice(filename)

            self.current_recording_id += 1
            self.recording_frames = []

    def start(self):
        print(f"🎧 Audio listener started on device: {self.input_device}")
        print(f"📊 Block size: {self.block_duration * 1000:.0f}ms ({self.block_size} samples)")
        try:
            with sd.InputStream(device=self.input_device,
                                channels=1,
                                samplerate=self.samplerate,
                                callback=self.audio_callback,
                                blocksize=self.block_size):
                while self.running:
                    sd.sleep(100)
        except KeyboardInterrupt:
            print("\nListener stopped by user")
        except Exception as e:
            print("Error in audio stream:", e)
        finally:
            self.stop()

    def stop(self):
        self.running = False
        self._stop_recording()
        print("Audio listener stopped")


def main():
    # Load voice signatures before starting
    load_voice_signatures()
    print(f"Loaded {len([v for v in VOICE_SIGNATURES if v is not None])} voice signatures")

    listener = AudioListener(
        input_device=INPUT_DEVICE,
        samplerate=SAMPLE_RATE,
        output_dir=OUTPUT_DIR
    )
    listener_thread = threading.Thread(target=listener.start)
    listener_thread.daemon = True
    listener_thread.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
        listener.stop()


if __name__ == "__main__":
    main()
