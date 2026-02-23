import os
import time

import numpy as np
import scipy.signal as sig
import soundfile as sf
import wave

from scipy import signal

from ntimit.consts import SAMPLE_RATE, BLOCK_SIZE, VOICE_SIGNATURES, voices, SIMILARITY_THRESHOLD, DEBOUNCE_SECONDS


def load_wav(path, target_fs=8000):
    x, fs = sf.read(path)
    if x.ndim > 1:
        x = x[:, 0]

    x = x.astype(np.float64)

    if fs != target_fs:
        x = sig.resample_poly(x, target_fs, fs)
        fs = target_fs

    x /= np.max(np.abs(x)) + 1e-9
    return fs, x

def frame_signal(x, fs, frame_ms=20):
    frame_len = int(fs * frame_ms / 1000)
    num_frames = len(x) // frame_len

    if num_frames == 0:
        # Return a single frame padded with zeros if needed
        frames = np.zeros((1, frame_len))
        frames[0, :len(x)] = x
        return frames

    frames = np.reshape(
        x[:num_frames * frame_len],
        (num_frames, frame_len)
    )

    return frames


def number_of_frames(frames):
    return frames.shape[0]


def bitrate_pcm(fs, bit_depth=16):
    return fs * bit_depth  # bits per second


def frame_fft(frame, fs):
    spectrum = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(len(frame), d=1 / fs)
    return freqs, spectrum


def effective_bandwidth(frame, fs, energy_ratio=0.95):
    freqs, spec = frame_fft(frame, fs)
    energy = spec ** 2
    total_energy = np.sum(energy)

    if total_energy == 0:
        return 0  # Return 0 bandwidth for silent frame

    cum_energy = np.cumsum(energy) / total_energy
    cum_energy = np.atleast_1d(cum_energy)  # Ensure it's at least 1D

    indices = np.where(cum_energy >= energy_ratio)[0]

    if len(indices) > 0:
        idx = indices[0]
        return freqs[idx]
    else:
        return freqs[-1]  # Return max frequency if threshold never reached


def dominant_frequency(frame, fs):
    freqs, spec = frame_fft(frame, fs)
    return freqs[np.argmax(spec)]

def extract_frame_count(file):
    fs, x = load_wav(file)
    frames = frame_signal(x, fs)
    return len(frames)

def extract_bandwidth(x,fs):
    frames = frame_signal(x, fs)
    bandwidths= []
    for f in frames:
        bandwidths.append( effective_bandwidth(f, fs) )
    return bandwidths

def extract_voice_features(x, fs):
    frames = frame_signal(x, fs)
    features = {
        "num_frames": number_of_frames(frames),
        "bitrate_pcm": bitrate_pcm(fs),
        "dominant_freqs": [],
        "bandwidths": []
    }

    for f in frames:
        features["dominant_freqs"].append(
            dominant_frequency(f, fs)
        )
        features["bandwidths"].append(
            effective_bandwidth(f, fs)
        )

    return features


def mse(x, y):
    min_len = min(len(x), len(y))
    return np.mean((x[:min_len] - y[:min_len]) ** 2)


def snr(original, distorted):
    min_len = min(len(original), len(distorted))

    noise = original[:min_len] - distorted[:min_len]

    return 10 * np.log10(
        np.sum(original[:min_len] ** 2) /
        (np.sum(noise ** 2) + 1e-9))


def spectral_distortion(x, y, fs):
    fx = frame_signal(x, fs)
    fy = frame_signal(y, fs)

    min_frames = min(len(fx), len(fy))
    dist = []

    for i in range(min_frames):
        _, sx = frame_fft(fx[i], fs)
        _, sy = frame_fft(fy[i], fs)

        sx += 1e-9
        sy += 1e-9

        dist.append(
            np.mean(np.abs(20 * np.log10(sx / sy)))
        )

    return np.mean(dist)

def longest_no_distortion(ogpath, decpath):

    fs, original = load_wav(ogpath)

    features = extract_voice_features(original, fs)
    bandwidth_og = features["bandwidths"]


    fs, decoded = load_wav(decpath)
    features = extract_voice_features(decoded, fs)
    bandwidth_dec = features["bandwidths"]

    # print("Frames:", features["num_frames"])
    # print("PCM Bitrate:", features["bitrate_pcm"], "bps")
    #  print("Dominant Freq:", features["dominant_freqs"])
    #print("Bandwidth:", features["bandwidths"])
    c = 0

    for i in range(len(bandwidth_og)):
        if bandwidth_og[i] != bandwidth_dec[i]:
            if i - c > 10:
                 print(bandwidth_og[i] - bandwidth_dec[i], c, i - 1, i - c)
            c = i

    #    print("-" * 20)
    #    print("MSE:", mse(original, decoded))
    #    print("SNR (dB):", snr(original, decoded))
    #    print("Spectral Distortion (dB):", spectral_distortion(original, decoded, fs))
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


def decrypt_voice(voice):
    decoded = decode_recorded_audio_aligned(voice)
    print("HEX_DATA (raw):", decoded)
    i = 2
    out_chars = []
    lost = False
    if decoded[1] == decoded[2] == 16 :
        i = 3

    if len(decoded)%2==1:
        lost = True
    if len(decoded) > 2 :
        while i + 1 < len(decoded):
            a = decoded[i]
            b = decoded[i + 1]

            if isinstance(a, int) and isinstance(b, int) and 0 <= a < 16 and 0 <= b < 16:
                hex_pair = f"{a:x}{b:x}"
                lower = 'a'
                upper = 'z'
                if hex_pair == ' ':
                    out_chars.append(' ')
                    i += 2
                elif lower <= hex_pair <= upper:
                    try:
                        char = bytes.fromhex(hex_pair).decode('utf-8', errors='replace')
                    except Exception:
                        char = '?'
                    out_chars.append(char)
                    i += 2
                else:
                    if lost:
                        i+=1
                        lost = False
                    # else:
                    #     lost+=1
                    #     hex6 =  f"{6:x}{b:x}"
                    #     #hex7 = f"{7:x}{b:x}"
                    #     try:
                    #         a = bytes.fromhex(hex6).decode('utf-8', errors='replace')
                    #         out_chars.append(a)
                    #         i+=2
                    #     except Exception:
                    #         i+=1

            else:
                i += 1

        if out_chars:
            text = ''.join(out_chars)
            print("Decrypted text appended:", text)
        else:
            print("No valid hex pairs found to decrypt.")
    else:
        print("null decoded")


def concatenate_wav_files_wave(input_files, output_file):
    """
    Concatenate WAV files using the built-in wave module.
    Assumes all files have the same sample rate, sample width, and channels.
    """
    if not input_files:
        raise ValueError("No input files provided")

    # Read first file to get parameters
    with wave.open(voices[input_files[0]], 'rb') as first_wav:
        params = first_wav.getparams()

    # Verify all files have same parameters
    for idx in input_files[1:]:
        with wave.open(voices[idx], 'rb') as wav:
            if wav.getparams() != params:
                # Check if it's just the number of frames that's different
                current_params = wav.getparams()
                if (current_params.nchannels != params.nchannels or
                        current_params.sampwidth != params.sampwidth or
                        current_params.framerate != params.framerate):
                    raise ValueError(f"File {idx} has incompatible parameters")

    # Write concatenated output
    with wave.open(output_file, 'wb') as output:
        output.setparams(params)

        for idx in input_files:
            with wave.open(voices[idx], 'rb') as wav:
                frames = wav.readframes(wav.getnframes())
                output.writeframes(frames)

    print(f"Successfully concatenated {len(input_files)} files to {output_file}")


