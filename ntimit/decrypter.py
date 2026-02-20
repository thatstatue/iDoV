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
SAMPLE_RATE = 16000
BLOCK_SIZE_SECONDS = 0.02
HEX_DATA = []
CHAR_DATA = []


def hex_to_text(hex_str):
    return bytes.fromhex(hex_str).decode('utf-8')


def voice_to_hex(voice):
    max_similarity = 0
    max_similar = None

    for trigger in voices:
        similarity = calculate_similarity(voice, trigger)
        if max_similarity is None or similarity > max_similarity:
            max_similarity = similarity
            max_similar = voice

    return max_similar


def calculate_similarity(audio_segment, trigger_sig):
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


def dummy_listener():
    device = 'pulse'
    sr = 16000
    duration = 1000 * 15
    record_file = "recorded/REC_DELETE_LATER.wav"

    play_thread = Thread(target=play_wav, args=(play_file, device, False))
    play_thread.start()
    record_from_virtual_speaker(record_file, device, sr, duration)
    play_thread.join()


class AudioListener:
    def __init__(self, input_device=None, samplerate=16000, output_dir="recorded"):
        self.input_device = input_device
        self.samplerate = samplerate
        self.output_dir = output_dir

        # 20ms block size (at 16kHz = 320 samples)
        self.block_duration = BLOCK_SIZE_SECONDS  # 20ms
        self.block_size = int(self.samplerate * self.block_duration)

        os.makedirs(output_dir, exist_ok=True)

        # Recording state
        self.listening = False
        self.recording_frames = []
        self.silent_blocks_count = 0
        self.silence_threshold = 3  # Number of silent blocks before stopping
        self.energy_threshold = 0.001  # Threshold for silence detection
        self.current_recording_id = 0
        self.running = True

    def _is_silent(self, audio_chunk):
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        return rms < self.energy_threshold

    def _calculate_similarity(self, audio_segment, trigger_sig):
        if trigger_sig is None or len(audio_segment) < len(trigger_sig):
            return 0

        audio_segment = audio_segment / (np.max(np.abs(audio_segment)) + 1e-10)
        trigger_sig = trigger_sig / (np.max(np.abs(trigger_sig)) + 1e-10)

        # Cross-correlation with tolerance for shift/jitter
        # Use 'same' mode to allow for alignment at different positions
        correlation = np.correlate(audio_segment, trigger_sig, mode='same')

        # Calculate normalized cross-correlation for all shifts
        norm_factor = np.sqrt(np.sum(trigger_sig ** 2) * np.sum(audio_segment ** 2))

        # Find maximum correlation across all shifts (tolerates shift/jitter)
        similarity = np.max(np.abs(correlation)) / (norm_factor + 1e-10)

        return similarity

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")

        indata = indata.flatten()
        is_silent = self._is_silent(indata)

        # Process voice to hex conversion for non-silent blocks
        if not is_silent:
            audio_chunk = indata.copy()
            hex_chunk = voice_to_hex(audio_chunk)
            if hex_chunk:  # Only append if we got a valid hex value
                print("hex is: ", hex_chunk)
                HEX_DATA.append(hex_chunk)
                try:
                    char_chunk = hex_to_text(hex_chunk)
                    print("character is: ", char_chunk)
                    CHAR_DATA.append(char_chunk)
                except:
                    print("Could not decode hex to text")
            if not self.listening:
                if not is_silent:
                    print(f"\n🎤 Sound detected! Starting recording at {time.inputBufferAdcTime:.2f}s")
                    self.listening = True
                    self.silent_blocks_count = 0
                    self.recording_frames = [audio_chunk]
            else:
                self.recording_frames.append(audio_chunk)
        else:
            if self.listening:
                self.silent_blocks_count += 1
                if self.silent_blocks_count >= self.silence_threshold:
                    print(f"\n🛑 {self.silence_threshold} consecutive silent blocks detected. Stopping recording.")
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

            self.decrypt_voice()

            self.current_recording_id += 1
            self.recording_frames = []

    def decrypt_voice(self):
        print("Decrypted HEX data:", HEX_DATA)
        print("Decrypted CHAR data:", CHAR_DATA)

    def start(self):
        """Start the audio listener"""
        print(f"🎧 Audio listener started on device: {self.input_device}")
        print(f"📊 Block size: {self.block_duration * 1000:.0f}ms ({self.block_size} samples)")
        print(f"🔇 Silence threshold: {self.silence_threshold} consecutive blocks")
        print("👂 Waiting for sound to start recording...")
        print("Press Ctrl+C to stop the listener\n")

        try:
            with sd.InputStream(
                    device=self.input_device,
                    channels=1,
                    samplerate=self.samplerate,
                    callback=self.audio_callback,
                    blocksize=self.block_size  # Process in 20ms blocks
            ):
                while self.running:
                    sd.sleep(100)

        except KeyboardInterrupt:
            print("\n\n🛑 Listener stopped by user")
        except Exception as e:
            print(f"Error in audio stream: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the audio listener"""
        self.running = False
        if self.listening:
            self._stop_recording()
        print("Audio listener stopped")


def main():
    listener = AudioListener(
        input_device='pulse',
        samplerate=SAMPLE_RATE,
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