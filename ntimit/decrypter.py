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
SILENCE_THRESHOLD = 150
# Load voice signatures at module level
VOICE_SIGNATURES = []
SAMPLE_RATE = 16000


def load_voice_signatures():
    """Load all voice files as numpy arrays"""
    global VOICE_SIGNATURES
    VOICE_SIGNATURES = []

    for voice_file in voices:
        try:
            if os.path.exists(voice_file):
                data, sr = sf.read(voice_file, dtype='float32')

                # Convert to mono if stereo
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)

                # Resample if needed
                if sr != SAMPLE_RATE:
                    from scipy import signal
                    number_of_samples = int(len(data) * SAMPLE_RATE / sr)
                    data = signal.resample(data, number_of_samples)

                # Normalize
                data = data / (np.max(np.abs(data)) + 1e-10)
                VOICE_SIGNATURES.append(data)
            else:
                print(f"Warning: Voice file not found: {voice_file}")
                VOICE_SIGNATURES.append(None)
        except Exception as e:
            print(f"Error loading voice file {voice_file}: {e}")
            VOICE_SIGNATURES.append(None)


def hex_to_text(hex_str):
    try:
        return bytes.fromhex(hex_str).decode('utf-8')
    except:
        return "?"


def voice_to_hex(audio_chunk):
    """Match audio chunk to a voice signature and return corresponding hex"""
    global VOICE_SIGNATURES

    print(
        f"\n[DEBUG] voice_to_hex called with audio_chunk length: {len(audio_chunk) if audio_chunk is not None else 'None'}")
    print(f"[DEBUG] Number of voice signatures available: {len(VOICE_SIGNATURES)}")

    max_similarity = 0
    best_match_index = -1

    for i, trigger in enumerate(VOICE_SIGNATURES):
        print(f"\n[DEBUG] Checking signature {i}: length={len(trigger) if trigger is not None else 'None'}")

        if trigger is None:
            print(f"[DEBUG] Signature {i} is None, skipping")
            continue

        if len(audio_chunk) < len(trigger):
            print(
                f"[DEBUG] Audio chunk length ({len(audio_chunk)}) < trigger length ({len(trigger)}), skipping signature {i}")
            continue

        similarity = calculate_similarity(audio_chunk, trigger)
        print(f"[DEBUG] Similarity for signature {i}: {similarity:.4f}")

        if similarity > max_similarity:
            print(
                f"[DEBUG] New best match: signature {i} with similarity {similarity:.4f} (previous best: {max_similarity:.4f})")
            max_similarity = similarity
            best_match_index = i

    # Set a threshold for what counts as a match
    print(f"\n[DEBUG] Final max_similarity: {max_similarity:.4f}, best_match_index: {best_match_index}")

    if max_similarity > 0.5 and best_match_index >= 0:
        hex_result = format(best_match_index, 'x')
        print(
            f"[DEBUG] MATCH FOUND! Index {best_match_index} -> hex '{hex_result}' with similarity {max_similarity:.4f}")
        return hex_result

    if best_match_index >= 0:
        print(f"[DEBUG] No match: best similarity {max_similarity:.4f} below threshold 0.5")
    else:
        print("[DEBUG] No valid matches found")

    return None


def calculate_similarity(audio_segment, trigger_sig):
    """Calculate similarity between audio segment and trigger signature"""
    print(f"\n[DEBUG] calculate_similarity called")

    if trigger_sig is None:
        print("[DEBUG] Trigger signature is None, returning 0")
        return 0

    if len(audio_segment) < len(trigger_sig):
        print(f"[DEBUG] Audio segment length ({len(audio_segment)}) < trigger length ({len(trigger_sig)}), returning 0")
        return 0

    # Ensure both are numpy arrays of float type
    original_audio_len = len(audio_segment)
    original_trigger_len = len(trigger_sig)

    audio_segment = np.asarray(audio_segment, dtype=np.float32)
    trigger_sig = np.asarray(trigger_sig, dtype=np.float32)

    print(f"[DEBUG] Converted to numpy arrays: audio shape={audio_segment.shape}, trigger shape={trigger_sig.shape}")

    # Normalize both signals
    audio_max = np.max(np.abs(audio_segment))
    if audio_max > 0:
        audio_segment = audio_segment / audio_max
        print(f"[DEBUG] Normalized audio with max value: {audio_max:.4f}")
    else:
        print("[DEBUG] Audio max is 0, skipping normalization")

    trigger_max = np.max(np.abs(trigger_sig))
    if trigger_max > 0:
        trigger_sig = trigger_sig / trigger_max
        print(f"[DEBUG] Normalized trigger with max value: {trigger_max:.4f}")
    else:
        print("[DEBUG] Trigger max is 0, skipping normalization")

    # Cross-correlation
    try:
        correlation = np.correlate(audio_segment, trigger_sig, mode='valid')
        print(f"[DEBUG] Correlation result shape: {correlation.shape}, max correlation: {np.max(correlation):.4f}")

        # Normalize correlation
        norm_factor = np.sqrt(np.sum(trigger_sig ** 2) * np.sum(audio_segment[:len(trigger_sig)] ** 2))
        print(f"[DEBUG] Normalization factor: {norm_factor:.4f}")

        if norm_factor > 0:
            similarity = np.max(correlation) / norm_factor
            print(f"[DEBUG] Calculated similarity: {similarity:.4f}")
        else:
            similarity = 0
            print("[DEBUG] Norm factor is 0, similarity set to 0")

        return similarity
    except Exception as e:
        print(f"[DEBUG] Error in correlation: {e}")
        return 0

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
        self.block_size = int(self.samplerate * self.block_duration)*4

        os.makedirs(output_dir, exist_ok=True)

        # Recording state
        self.listening = False
        self.recording_frames = []
        self.silent_blocks_count = 0
        self.silence_threshold = SILENCE_THRESHOLD
        self.energy_threshold = 0.001
        self.current_recording_id = 0
        self.running = True

        # Buffer for accumulating audio for better voice recognition
        self.accumulation_buffer = []
        self.buffer_max_size = 5  # Accumulate up to 5 blocks for better matching

    def _is_silent(self, audio_chunk):
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        return rms < self.energy_threshold

    def _calculate_similarity(self, audio_segment, trigger_sig):
        # This is now handled by the global function
        return calculate_similarity(audio_segment, trigger_sig)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")

        indata = indata.flatten()
        is_silent = self._is_silent(indata)

        # Process voice to hex conversion for non-silent blocks
        if not is_silent:
            audio_chunk = indata.copy()
            self.decrypt_chunk(audio_chunk)
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

    def decrypt_chunk(self, audio_chunk):
        # Add to accumulation buffer
        self.accumulation_buffer.append(audio_chunk)
        if len(self.accumulation_buffer) > self.buffer_max_size:
            self.accumulation_buffer.pop(0)

        # Use accumulated audio for better recognition
        if len(self.accumulation_buffer) >= 3:  # Need at least 3 blocks
            accumulated_audio = np.concatenate(self.accumulation_buffer)
            hex_chunk = voice_to_hex(accumulated_audio)

            if hex_chunk:  # Only append if we got a valid hex value
                print("hex is: ", hex_chunk)
                HEX_DATA.append(hex_chunk)
                try:
                    char_chunk = hex_to_text(hex_chunk)
                    print("character is: ", char_chunk)
                    CHAR_DATA.append(char_chunk)
                except:
                    print("Could not decode hex to text")

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
    # Load voice signatures before starting
    global SAMPLE_RATE
    SAMPLE_RATE = 16000
    load_voice_signatures()
    print(f"Loaded {len([v for v in VOICE_SIGNATURES if v is not None])} voice signatures")

    listener = AudioListener(
        input_device='pulse',
        samplerate=16000,
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