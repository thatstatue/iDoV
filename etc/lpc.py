

import numpy as np

import scipy.signal as sig

import scipy.io.wavfile as wav

def read_wav_mono(path, target_fs=8000):

    fs, x = wav.read(path)

    if x.ndim > 1:

        x = x[:, 0]

    x = x.astype(np.float64)

    if fs != target_fs:

        x = sig.resample_poly(x, target_fs, fs)

        fs = target_fs

    x /= np.max(np.abs(x)) + 1e-9

    return fs, x


def frame_signal(x, fs, frame_ms=25, hop_ms=10):

    frame_len = int(fs * frame_ms / 1000)

    hop_len = int(fs * hop_ms / 1000)

    frames = []

    for i in range(0, len(x) - frame_len, hop_len):

        frames.append(x[i:i+frame_len])

    return np.array(frames)



def lpc_analysis(frame, order=10):

    frame = frame * np.hamming(len(frame))

    a = sig.lpc(frame, order)

    return a


# residual = inverse filter خروجی

def lpc_residual(frame, lpc_coeffs):

    residual = sig.lfilter(lpc_coeffs, [1.0], frame)

    return residual


def extract_residuals(frames, order=10):

    residuals = []

    envelopes = []

    for f in frames:

        a = lpc_analysis(f, order)

        r = lpc_residual(f, a)

        # energy normalize

        r /= np.sqrt(np.mean(r**2) + 1e-9)

        residuals.append(r)

        envelopes.append(a)

    return np.array(residuals), np.array(envelopes)


def shuffle_residuals(residuals):

    idx = np.random.permutation(len(residuals))

    return residuals[idx]


def average_envelope(envelopes):

    return np.mean(envelopes, axis=0)


def synthesize_signal(residuals, lpc_env):

    out = []

    for r in residuals:

        s = sig.lfilter([1.0], lpc_env, r)

        out.append(s)

    y = np.concatenate(out)

    y /= np.max(np.abs(y)) + 1e-9

    return y

def build_robust_test_signal(wav_path):

    fs, x = read_wav_mono(wav_path)

    frames = frame_signal(x, fs)

    residuals, envelopes = extract_residuals(frames, order=10)

    residuals = shuffle_residuals(residuals)

    avg_env = average_envelope(envelopes)

    y = synthesize_signal(residuals, avg_env)

    return fs, y

if __name__ == '__main__':
    fs, test_signal = build_robust_test_signal("../ntimit/OG/SA1.WAV")
    wav.write("robust_non_intelligible.wav", fs, test_signal.astype(np.float32))

