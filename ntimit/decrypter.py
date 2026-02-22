from ntimit.encrypter import voices
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
import os
from scipy import signal

buffer = "recorded/buffer.WAV"
device = 'pulse'
SAMPLE_RATE = 16000
BLOCK_SIZE_SECONDS = 0.24
#HEX_DATA = []
CHAR_DATA = []
SILENCE_THRESHOLD = 1
VOICE_SIGNATURES = []
SIMILARITY_THRESHOLD = 0

BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_SIZE_SECONDS)  # 3840
DEBOUNCE_SECONDS = 0.20  # ignore detections within this time window


def load_voice_signatures():
    """
    Load each voice file, resample to SAMPLE_RATE, and force length == BLOCK_SIZE
    by center-cropping or zero-padding. Normalize to zero-mean, unit-std.
    """
    global VOICE_SIGNATURES
    VOICE_SIGNATURES = []

    for idx, voice_file in enumerate(voices):
        try:
            if not os.path.exists(voice_file):
                print(f"Warning: Voice file not found: {voice_file}")
                VOICE_SIGNATURES.append(None)
                continue

            data, sr = sf.read(voice_file, dtype='float32')
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            # Resample if needed
            if sr != SAMPLE_RATE:
                num_samples = int(len(data) * SAMPLE_RATE / sr)
                data = signal.resample(data, num_samples)

            # Normalize amplitude (avoid division by zero)
            max_abs = np.max(np.abs(data)) + 1e-10
            data = data / max_abs

            # Force exact length BLOCK_SIZE: center-crop or pad with zeros
            if len(data) > BLOCK_SIZE:
                start = (len(data) - BLOCK_SIZE) // 2
                data = data[start:start + BLOCK_SIZE]
            elif len(data) < BLOCK_SIZE:
                pad = BLOCK_SIZE - len(data)
                left = pad // 2
                right = pad - left
                data = np.pad(data, (left, right), mode='constant', constant_values=0.0)

            # Remove DC and normalize to unit-std (for Pearson)
            data = data - np.mean(data)
            std = np.std(data) + 1e-10
            data = data / std

            VOICE_SIGNATURES.append(data.astype(np.float32))
        except Exception as e:
            print(f"Error loading voice file {voice_file}: {e}")
            VOICE_SIGNATURES.append(None)


def hex_to_text(hex_str):
    try:
        v = bytes.fromhex(hex_str).decode('utf-8')

        return v
    except:
        return "?"


def voice_to_hex(audio_chunk , debounce=False):
    """
    Match a single BLOCK-sized audio chunk to the best voice signature.
    Returns the index (int) of matched voice in voices list, or None if ambiguous/low-score.
    """
    global VOICE_SIGNATURES, _last_detection_time, _last_detection_index

    if audio_chunk is None or len(audio_chunk) != BLOCK_SIZE:
        # If it's not the exact length, reject
        print(f"[DEBUG] voice_to_hex: rejecting chunk of length {None if audio_chunk is None else len(audio_chunk)}")
        return None

    # Preprocess audio chunk: DC removal and unit-std
    a = np.asarray(audio_chunk, dtype=np.float32)
    a = a - np.mean(a)
    std_a = np.std(a) + 1e-10
    a = a / std_a

    best_idx = None
    best_score = -2.0
    second_best = -2.0
    scores = []

    for i, trig in enumerate(VOICE_SIGNATURES):
        if trig is None:
            continue
        score = calculate_similarity_fast(a, trig)
        scores.append((i, score))
        if score > best_score:
            second_best = best_score
            best_score = score
            best_idx = i
        elif score > second_best:
            second_best = score

    # Debug print top few scores
    scores.sort(key=lambda x: x[1], reverse=True)
    if len(scores) > 0:
        top_debug = ", ".join([f"{i}:{s:.3f}" for i, s in scores[:5]])
        print(f"[DEBUG] top scores: {top_debug}")

    # Enforce threshold and margin
    if best_score >= SIMILARITY_THRESHOLD:
        now = time.time()
        # Debounce: ignore immediate repeats within DEBOUNCE_SECONDS
        if debounce and _last_detection_index == best_idx and (now - _last_detection_time) < DEBOUNCE_SECONDS:
            print(f"[DEBUG] Debounced repeated detection idx={best_idx}")
            return None
        _last_detection_time = now
        _last_detection_index = best_idx
        print(f"[DEBUG] Matched idx={best_idx} score={best_score:.3f} margin={best_score - second_best:.3f}")
        return best_idx

    print(f"[DEBUG] No clear match. best={best_score:.3f}, second={second_best:.3f}")
    return None


def calculate_similarity_fast(audio_segment, trigger_sig):
    """
    Phase-insensitive similarity using FFT magnitude correlation.
    """

    if trigger_sig is None:
        return -2.0

    # Windowing improves stability
    window = np.hanning(len(audio_segment))

    a = audio_segment * window
    t = trigger_sig * window

    # FFT
    A = np.abs(np.fft.rfft(a))
    T = np.abs(np.fft.rfft(t))

    # Normalize
    A = A - np.mean(A)
    T = T - np.mean(T)

    A /= (np.std(A) + 1e-10)
    T /= (np.std(T) + 1e-10)

    return float(np.dot(A, T) / len(T))

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
        """Ensure block is length block_size, then add to queue for processing."""
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

    def _process_full_buffer(self):
        while len(self._sample_buffer) >= self.block_size:
            chunk = self._sample_buffer[:self.block_size]
            self._sample_buffer = self._sample_buffer[self.block_size:]

            detected = voice_to_hex(chunk)

            if detected is not None:
                #HEX_DATA.append(detected) todo fix
                print(f"[DETECT] appended token {detected}")
            else:
                print("[DETECT] no confident token for this block")

    def audio_callback(self, indata, frames, time_, status):
        if status: print("Audio callback status:", status)
        is_silent = self._is_silent(indata)
        if not is_silent:
            self.listening = True
            self.silent_blocks_count = 0

            print("not silent")
            samples = indata.flatten().astype(np.float32)
            self._sample_buffer = np.concatenate((self._sample_buffer, samples))
            self.recording_frames.append(indata.copy())
        else:  # silent block

            if self.listening:
                print("listening")
                self.silent_blocks_count += 1
                if self.silent_blocks_count >= self.silence_threshold:
                    print("more")
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
           # decode_recorded_audio_aligned(audio_np)
            decoded = decode_recorded_audio_aligned(filename)
            # After saving the file, attempt decryption of token stream
            self.decrypt_voice(decoded)

            self.current_recording_id += 1
            self.recording_frames = []
        #self._process_full_buffer() todo might add

    def decrypt_voice(self, decoded):
        # Build bytes from pairs of nibbles (0..15). Values >=16 are treated as separators/control.
        print("HEX_DATA (raw):", decoded)
        i = 0
        out_chars = []
        while not decoded[i]==16:
            print("HEX_DATA (raw):", decoded[i])
            i+=1
        while i + 1 < len(decoded):
            a = decoded[i]
            b = decoded[i + 1]
            # only combine if both are integer nibbles 0..15
            if isinstance(a, int) and isinstance(b, int) and 0 <= a < 16 and 0 <= b < 16:
                # convert to hex string: 0..9 -> '0'..'9', 10..15 -> 'a'..'f'
                hex_pair = f"{a:x}{b:x}"
                try:
                    # decode pair into a single byte then to utf-8 char
                    char = bytes.fromhex(hex_pair).decode('utf-8', errors='replace')
                except Exception:
                    char = '?'
                out_chars.append(char)
                i += 2
            else:
                # skip single token if it's a separator or control or out-of-range
                i += 1

        # append to CHAR_DATA
        if out_chars:
            text = ''.join(out_chars)
            CHAR_DATA.append(text)
            print("Decrypted text appended:", text)
        else:
            print("No valid hex pairs found to decrypt.")
        # clear HEX_DATA after processing to avoid re-processing same tokens
        #HEX_DATA.clear()

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

def decode_recorded_audio_aligned(audio_input, step=40):
    if isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise FileNotFoundError(f"File not found: {audio_input}")

        audio, sr = sf.read(audio_input, dtype='float32')

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if sr != SAMPLE_RATE:
            num_samples = int(len(audio) * SAMPLE_RATE / sr)
            audio = signal.resample(audio, num_samples)

    else:
        audio = np.asarray(audio_input, dtype=np.float32)
    # normalize once globally (helps stability)
    audio = audio - np.mean(audio)
    audio = audio / (np.std(audio) + 1e-10)

    total_len = len(audio)
    best_offset = 0
    best_avg_score = -1.0

    print("[ALIGN] Searching best starting offset...")

    # Try candidate offsets only within first BLOCK_SIZE
    for offset in range(0, BLOCK_SIZE, step):

        scores = []
        pos = offset

        while pos + BLOCK_SIZE <= total_len:
            chunk = audio[pos:pos + BLOCK_SIZE]
            idx, score = _best_match_score(chunk)
            if score is not None:
                scores.append(score)
            pos += BLOCK_SIZE

        if len(scores) == 0:
            continue

        avg_score = np.mean(scores)

        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_offset = offset

    print(f"[ALIGN] Best offset = {best_offset} samples "
          f"(~{best_offset/SAMPLE_RATE:.4f}s) "
          f"avg_score={best_avg_score:.3f}")

    # Now decode using best offset
    decoded = []
    pos = best_offset

    while pos + BLOCK_SIZE <= total_len:
        chunk = audio[pos:pos + BLOCK_SIZE]
        token = voice_to_hex(chunk)
        if token is not None:
            decoded.append(token)
        pos += BLOCK_SIZE

    return decoded

def _best_match_score(chunk):
    """
    Returns (best_index, best_score) without threshold rejection.
    Used only for alignment scoring.
    """

    if len(chunk) != BLOCK_SIZE:
        return None, None

    a = chunk - np.mean(chunk)
    a /= (np.std(a) + 1e-10)

    best_idx = None
    best_score = -2.0

    for i, trig in enumerate(VOICE_SIGNATURES):
        if trig is None:
            continue
        score = float(np.dot(a, trig) / len(trig))
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx, best_score


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
