from xmlrpc.client import MAXINT
#todo twinkle speaker has a problem the rest is fixed
import numpy as np
from scipy.io.wavfile import read
# تنظیمات
FS = 8000
SAMPLES_PER_FRAME = 160
test_min_power_distance = 0.001
COUNT = 10
FREQ_DISTANCE = 1000 / COUNT
FREQ_0 = 1200
BAUD_RATE = 50

def read_wav(name, frame_size=1024, hop_size=1024):
    fs, audio = read(name)

    if audio.ndim > 1:
        audio = audio[:, 0]

    audio = audio.astype(float)

    freqs = np.fft.rfftfreq(frame_size, 1 / fs)


    for start in range(0, len(audio) - frame_size, hop_size):
        frame = audio[start:start + frame_size]

        fft_vals = np.abs(np.fft.rfft(frame))
        power = [0,0,0,0,0,
                 0,0,0,0,0]
        max_power = 0
        min_power = MAXINT
        i = 0;
        for index in range(0, len(power)):
            FREQ_INDEX = FREQ_0 + index* FREQ_DISTANCE
            idx = np.argmin(np.abs(freqs - FREQ_INDEX))
            power[index] = fft_vals[idx]
            if power[index]>max_power:
                max_power = max(max_power, power[index])
                i = index
            min_power = min(min_power, power[index])

        detected_bit = None
        if max_power- min_power >= test_min_power_distance:
             detected_bit = i
        if detected_bit is not None:
            print(detected_bit , " , ")


if __name__ == "__main__":
    read_wav("tx_output_debug.wav")