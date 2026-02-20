
from ntimit.encrypter import voices
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
import os

buffer = "recorded/buffer.WAV"
device = 'pulse'
SAMPLE_RATE = 16000
BLOCK_SIZE_SECONDS = 0.24
HEX_DATA = []
CHAR_DATA = []
SILENCE_THRESHOLD = 10
VOICE_SIGNATURES = []


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
        v = bytes.fromhex(hex_str).decode('utf-8')

        return v
    except:
        return "?"


def voice_to_hex(audio_chunk):
    """Match audio chunk to a voice signature using Pearson correlation"""
    global VOICE_SIGNATURES

    print(
        f"\n[DEBUG] voice_to_hex called with audio_chunk length: {len(audio_chunk) if audio_chunk is not None else 'None'}")

    max_similarity = -1  # شروع با -1 چون پیرسون می‌تواند منفی باشد
    best_match_index = -1
    similarities = []

    for i, trigger in enumerate(VOICE_SIGNATURES):
        if trigger is None or len(audio_chunk) < len(trigger):
            continue

        similarity = calculate_similarity_fast(audio_chunk, trigger)  # استفاده از روش سریع
        similarities.append((i, similarity))
        print(f"[DEBUG] Signature {i}: similarity = {similarity:.4f}")

        if similarity > max_similarity:
            max_similarity = similarity
            best_match_index = i

    # آستانه مناسب برای پیرسون (بعد از تبدیل به 0-1)
    threshold = 0.6  # می‌توانید تنظیم کنید

    print(f"\n[DEBUG] Best match: index {best_match_index}, similarity {max_similarity:.4f}")

    # محاسبه فاصله از دومین شباهت بزرگ (برای اطمینان بیشتر)
    if len(similarities) > 1:
        similarities.sort(key=lambda x: x[1], reverse=True)
        if similarities[0][1] > threshold and similarities[0][1] - similarities[1][1] > 0.1:
            print(f"[DEBUG] Clear match with margin: {similarities[0][1] - similarities[1][1]:.4f}")
            return similarities[0][0]

    if max_similarity > threshold and best_match_index >= 0:
        return best_match_index

    return None

def calculate_similarity_fast(audio_segment, trigger_sig):
    """Fast similarity using FFT-based cross-correlation"""
    print(f"\n[DEBUG] calculate_similarity_fast called")

    if trigger_sig is None:
        return 0

    if len(audio_segment) < len(trigger_sig):
        return 0

    # Convert to numpy arrays
    audio = np.asarray(audio_segment, dtype=np.float32)
    trigger = np.asarray(trigger_sig, dtype=np.float32)

    # Normalize both signals to zero mean and unit variance
    audio_norm = (audio - np.mean(audio)) / (np.std(audio) + 1e-10)
    trigger_norm = (trigger - np.mean(trigger)) / (np.std(trigger) + 1e-10)

    # Use FFT for fast cross-correlation
    # Pad to next power of 2 for FFT efficiency
    n = len(audio) + len(trigger) - 1
    n_fft = 1 << (n - 1).bit_length()

    # Compute FFT of both signals
    audio_fft = np.fft.rfft(audio_norm, n=n_fft)
    trigger_fft = np.fft.rfft(trigger_norm, n=n_fft)

    # Cross-correlation in frequency domain
    correlation = np.fft.irfft(audio_fft * np.conj(trigger_fft), n=n_fft)

    # Take only valid part
    correlation = correlation[:len(audio) - len(trigger) + 1]

    # Normalize correlation to [-1, 1] range
    max_corr = np.max(correlation) / len(trigger)

    # Convert to similarity in [0, 1] range
    similarity = (max_corr + 1) / 2

    print(f"[DEBUG] FFT method - Max correlation: {max_corr:.4f}")
    print(f"[DEBUG] FFT method - Similarity: {similarity:.4f}")

    return similarity


class AudioListener:
    def __init__(self, input_device=None, samplerate=16000, output_dir="recorded"):
        self.input_device = input_device
        self.samplerate = samplerate
        self.output_dir = output_dir
        self.block_duration = BLOCK_SIZE_SECONDS  # 20 * 12 ms = 240 ms
        self.block_size = int(self.samplerate * self.block_duration)

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
        i = 0
        if len(HEX_DATA)>1:
            while i+1< len(HEX_DATA):
                if HEX_DATA[i]<16 and HEX_DATA[i+1]<16:
                    if ( HEX_DATA[i]>9):
                        HEX_DATA[i] = 'a' + (HEX_DATA[i]-10)
                    if (HEX_DATA[i+1] > 9):
                        HEX_DATA[i+1] = 'a' + (HEX_DATA[i+1] - 10)

                    hex_str = str(HEX_DATA[i])+""+str(HEX_DATA[i+1])
                    print("hex string : ",hex_str)
                    text = hex_to_text(hex_str)
                    i+=2
                    CHAR_DATA.append(text)
                    print("Decrypted hex:", text)
                else:
                    print("#")
                    i += 1
        else:
            print("Hex Data is empty")
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
                    blocksize=self.block_size
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