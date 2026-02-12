import numpy as np
import scipy.signal as sig
import soundfile as sf

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
