import scipy.signal as sig
import numpy as np
import sounddevice as sd
import soundfile as sf

from ntimit.frcodec import run_vocoder_simulation


# sa2: fr
# 148 -> 172   25
# 188 -> 213         26
# 215 -> 233         19

# SA1-3: 115 163 49
# sa2-2:
# 97 117 21
# 124 144 21

# SI648 134 157 24
# SI943 94 116 23
# SI1271 207 240 34
# SI1406 101 121 21


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
    if np.sum(energy)==0:
        cum_energy = 0
    else:
        cum_energy = np.cumsum(energy) / np.sum(energy)

    idx = np.where(cum_energy >= energy_ratio)[0][0]
    return freqs[idx]


def dominant_frequency(frame, fs):
    freqs, spec = frame_fft(frame, fs)
    return freqs[np.argmax(spec)]


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


def slice_wav_by_frames(input_wav, output_wav, start_frame, end_frame, frame_ms=20):
    data, sr = sf.read(input_wav)
    if data.ndim > 1:
        data = data[:, 0]  # mono
    samples_per_frame = int(sr * frame_ms / 1000)
    start_sample = start_frame * samples_per_frame
    end_sample = end_frame * samples_per_frame
    sliced = data[start_sample:end_sample]
    sf.write(output_wav, sliced, sr)



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

def comparator(ogpath, decpath):

    fs, original = load_wav(ogpath)

    features = extract_voice_features(original, fs)
    bandwidth_og = features["bandwidths"]
    fs, decoded = load_wav(decpath)
    features = extract_voice_features(decoded, fs)
    bandwidth_dec = features["bandwidths"]

    t = 0
    for i in range(len(bandwidth_og)):
        if bandwidth_og[i] != bandwidth_dec[i]:
            print("diff in ", ogpath, i)
            t = t +1
    if t<4 :
        print( "diffs in ", ogpath, " are ", t)

def create_slices(name, s, f, d, c): #fin, diff, count
    l = (f-s)-d*(c-1)
    if l<1:
         print("length is too short")
    else:
        for i in range (c):
           out = name+"_WINNER_"+str(i)+"_"+str(l)+".WAV"
           print(out)
           slice_wav_by_frames(name+".WAV", out , s, s+l)
           s=s+d
    return l


def test_slices(name,start,finish,l):

    while start+l<=finish:
        inp = name+"_" + str(start) +".wav"
        out = name+"_" + str(start) +"_FR.wav"
        comparator(inp, out)
        start =start+l

def test_slices_WAV(name,start,finish,l):

    while start+l<=finish:
        inp = name+"_" + str(start) +".WAV"
        out = name+"_" + str(start) +"_FR.WAV"
        comparator(inp, out)
        start =start+l


def vocode_slices_WAV(name,start,finish,l):

    while start+l<=finish:  # ran/frame_count
        inp = name+"_" + str(start) +".WAV"
        out = name+"_" + str(start) +"_FR.WAV"
        run_vocoder_simulation(inp, out, 'exe/fr_enc.exe', 'exe/fr_dec.exe')
        start =start+l


def create_slices_length(name, s, f, length): #fin, diff, count
    c = int((f - s) / length)
    s = s+2
    if c<1:
         print("count is too short")
    else:
        for i in range (c):
           out = "eand.wav" #name +"_W_" + str(i) +"_" + str(length) + ".WAV"
           print(out)
           slice_wav_by_frames(name , out, s - 2, s + length)
           s= s + length
    return c
# SA1-3: 115 163 49
# sa2-2:
# 97 117 21
# 124 144 21

# SI648 134 157 24
# SI943 94 116 23
# SI1271 207 240 34
# SI1406 101 121 21
# sa2: fr
# 148 -> 172   25
# 188 -> 213         26
# 215 -> 233         19
def slice_up_WAV(fullinput, testhalf,start, l):
    t = start
    inp = testhalf+ str(t)+".WAV"
    slice_wav_by_frames(fullinput, inp, t, t+l)
    out = testhalf + str(t)+"_FR.WAV"
    run_vocoder_simulation(inp, out, 'exe/fr_enc.exe', 'exe/fr_dec.exe')

def test_winners():
    n = "test/SA1-3"
    # l = create_slices_length(n,148,172,12)
    # vocode_slices(n,16,l)
    test_slices(n, 115, 168, 12)
    n = "test/SA2-2"
    test_slices(n, 97, 110, 12)
    test_slices(n, 105, 128, 12)
    test_slices(n, 132, 150, 12)
    test_slices(n, 124, 136, 12)

    n = "test/SA2"
    # test_slices(n,148,175,12)
    # test_slices(n,188,230,12)
def test_winners_WAV(testhalf, start,finish,l):
    n = testhalf
    vocode_slices_WAV(n,start,finish,l)
    test_slices_WAV(n, start, finish, l)

def sliceandtest():
    l = 12
    testhalf = "test/SA1-3_"
    slice_up_WAV("SA1-3.WAV", testhalf, 115, l)
    slice_up_WAV("SA1-3.WAV", testhalf, 115 + l, l)
    slice_up_WAV("SA1-3.WAV", testhalf, 115 + l + l, l)
    slice_up_WAV("SA1-3.WAV", testhalf, 115 + l + l + l, l)

    test_winners_WAV("test/SA1-3", 115, 166, 12)
    # comparator("test/SA1-3_115.wav","recorded/REC_SA1-3_115.wav")


if __name__ == "__main__":
   comparator("test/SA1-3_115.WAV","recorded/REC_SA1-3_115.WAV")