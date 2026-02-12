from threading import Thread

from pygame.examples.music_drop_fade import play_file

from ntimit.encrypter import voices, voices_start, voices_finish
from ntimit.virtualmic import record_from_virtual_speaker, play_wav
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
import os
buffer = "recorded/buffer.WAV"
device = 'pulse'


def hex_to_text(hex):
    return bytes.fromhex(hex).decode('utf-8')

def voice_to_hex(voice):
    print("i dont know") #todo

def dummy_listener():
    device = 'pulse'
    sr = 16000
    duration = 1000 * 15
    record_file ="recorded/REC_DELETE_LATER.wav"

    play_thread = Thread(target=play_wav, args=(play_file, device, False))
    play_thread.start()
    record_from_virtual_speaker(record_file, device, sr, duration)
    play_thread.join()
    # Thread(listener(device)).start()

class AudioListener:
    def __init__(self, input_device=None, samplerate=16000, start_trigger_file=voices[voices_start]
                 ,stop_trigger_file = voices[voices_finish],output_dir="recorded"):
        self.input_device = input_device
        self.samplerate = samplerate
        self.start_trigger_file = start_trigger_file
        self.stop_trigger_file = stop_trigger_file
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)
        self.start_trigger_sig = self._load_audio_signature(start_trigger_file)
        self.stop_trigger_sig = self._load_audio_signature(stop_trigger_file)

        self.is_recording = False
        self.recording_frames = []
        self.current_recording_id = 0
        self.running = True

        self.buffer_duration = 2.0  # Buffer 2 seconds of audio
        self.buffer_size = int(self.samplerate * self.buffer_duration)
        self.audio_buffer = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.buffer_index = 0

        self.similarity_threshold = 539952535552  # Correlation threshold for trigger detection

    def _load_audio_signature(self, filepath):
        """Load and preprocess trigger audio signature"""
        if not os.path.exists(filepath):
            print(f"Warning: Trigger file {filepath} not found")
            return None

        # Load audio
        data, sr = sf.read(filepath, dtype='float32')

        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Resample if necessary
        if sr != self.samplerate:
            from scipy import signal
            number_of_samples = int(len(data) * self.samplerate / sr)
            data = signal.resample(data, number_of_samples)

        # Normalize
        data = data / (np.max(np.abs(data)) + 1e-10)

        return data

    def _calculate_similarity(self, audio_segment, trigger_sig):
        """Calculate similarity between audio segment and trigger signature"""
        if trigger_sig is None or len(audio_segment) < len(trigger_sig):
            return 0

        # Normalize both signals
        audio_segment = audio_segment / (np.max(np.abs(audio_segment)) + 1e-10)

        # Cross-correlation
        correlation = np.correlate(audio_segment, trigger_sig, mode='valid')

        # Normalize correlation
        norm_factor = np.sqrt(np.sum(trigger_sig ** 2) * np.sum(audio_segment[:len(trigger_sig)] ** 2))
        similarity = np.max(correlation) / (norm_factor + 1e-10)

        return similarity

    def _check_for_triggers(self, audio_chunk):
        """Check if the audio chunk contains start or stop triggers"""
        # Update circular buffer
        chunk_len = len(audio_chunk)
        if chunk_len >= self.buffer_size:
            # Chunk is larger than buffer, just use the end part
            self.audio_buffer = audio_chunk[-self.buffer_size:]
        else:
            # Shift buffer and add new chunk
            self.audio_buffer = np.roll(self.audio_buffer, -chunk_len, axis=0)
            self.audio_buffer[-chunk_len:] = audio_chunk

        # Check for triggers in the buffer
        if self.start_trigger_sig is not None:
            similarity = self._calculate_similarity(self.audio_buffer.flatten(),
                                                    self.start_trigger_sig)
            if similarity > self.similarity_threshold and not self.is_recording:
                print(f"\n🎤 Start trigger detected! (similarity: {similarity:.2f})")
                self._start_recording()

        if self.stop_trigger_sig is not None and self.is_recording:
            similarity = self._calculate_similarity(self.audio_buffer.flatten(),
                                                    self.stop_trigger_sig)
            if similarity > self.similarity_threshold:
                print(f"\n🛑 Stop trigger detected! (similarity: {similarity:.2f})")
                self._stop_recording()

    def _start_recording(self):
        """Start recording audio"""
        self.is_recording = True
        self.recording_frames = []
        # Include the buffer content that led to the trigger
        self.recording_frames.append(self.audio_buffer.copy())
        print(f"Recording started at {time.strftime('%H:%M:%S')}")

    def _stop_recording(self):
        """Stop recording and save the file"""
        self.is_recording = False

        if self.recording_frames:
            # Concatenate all recorded frames
            audio_np = np.concatenate(self.recording_frames, axis=0)

            # Save the recording
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"{self.output_dir}/recording_{timestamp}_{self.current_recording_id}.wav"
            sf.write(filename, audio_np, self.samplerate)

            duration = len(audio_np) / self.samplerate
            print(f"✅ Recording saved: {filename} (duration: {duration:.2f}s)")

            self.current_recording_id += 1
            self.recording_frames = []

    def audio_callback(self, indata, frames, time, status):
        """Callback function for the audio stream"""
        if status:
            print(f"Audio callback status: {status}")

        # Check for triggers in the incoming audio
        self._check_for_triggers(indata.copy())

        # If recording, save the audio
        if self.is_recording:
            self.recording_frames.append(indata.copy())

    def start(self):
        """Start the audio listener"""
        print(f"🎧 Audio listener started on device: {self.input_device}")
        print(f"👂 Waiting for '{self.start_trigger_file}' to start recording...")
        print(f"✋ Say '{self.stop_trigger_file}' to stop recording")
        print("Press Ctrl+C to stop the listener\n")

        try:
            with sd.InputStream(
                    device=self.input_device,
                    channels=1,  # Mono is sufficient for trigger detection
                    samplerate=self.samplerate,
                    callback=self.audio_callback,
                    blocksize=4096  # Process in smaller chunks for faster response
            ):
                while self.running:
                    sd.sleep(100)  # Sleep for 100ms

        except KeyboardInterrupt:
            print("\n\n🛑 Listener stopped by user")
        except Exception as e:
            print(f"Error in audio stream: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the audio listener"""
        self.running = False
        if self.is_recording:
            self._stop_recording()
        print("Audio listener stopped")


def main():
    listener = AudioListener(
        input_device='pulse',
        samplerate=16000,
        start_trigger_file=voices[voices_start],
        stop_trigger_file=voices[voices_finish],
        output_dir="recorded"
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
