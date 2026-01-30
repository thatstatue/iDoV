import numpy as np
import scipy.io.wavfile as wav
import time

FS = 8000  # نرخ نمونه‌برداری تلفن
FRAME_DURATION_MS = 20  # طول هر فریم GSM
SAMPLES_PER_FRAME = int(FS * (FRAME_DURATION_MS / 1000))
COUNT = 10
FREQ_DISTANCE = 1000 / COUNT
FREQ_0 = 1200
BAUD_RATE = 50
SAFETY_REPEAT = 6
LOOKUP_TABLE = [5,3,7,1,8,2,6,0,4,9]

class FrameBasedTransmitter:
    def __init__(self):
        self.buffer = np.array([], dtype=np.float32)
        self.phase_accumulator = 0

    def generate_frame(self, num):
        freq = FREQ_0 + num * FREQ_DISTANCE

        t = np.arange(SAMPLES_PER_FRAME) / FS

        phase = 2 * np.pi * freq * t + self.phase_accumulator  # (Phase Continuity)

        frame = 0.5 * np.sin(phase)

        # آپدیت فاز برای فریم بعدی
        self.phase_accumulator = phase[-1] + (2 * np.pi * freq * (1 / FS))

        return frame.astype(np.float32)


def run_simulation():
    tx = FrameBasedTransmitter()  # sends the max of 7 packets

    test_bits = [0,5,1,6,2,7,3,8,4,9,
                 0, 5, 1, 6, 2, 7, 3, 8, 4, 9,
                 0, 5, 1, 6, 2, 7, 3, 8, 4, 9,
                 0, 5, 1, 6, 2, 7, 3, 8, 4, 9,
                 0, 5, 1, 6, 2, 7, 3, 8, 4, 9,
                 0, 5, 1, 6, 2, 7, 3, 8, 4, 9,
                 0, 5, 1, 6, 2, 7, 3, 8, 4, 9,
                 0, 5, 1, 6, 2, 7, 3, 8, 4, 9,
                 0,5,1,6,2,7,3,8,4,9]
      #  [1, 0, 0, 1, 0, 1, 0,  1, 1, 1, 1, 1, 1, 1,
                 # 1, 1, 1, 1, 1, 1, 0,
                 # 0, 0, 0, 0, 0, 0, 0,
                 # 1, 1, 1, 1, 1, 1, 1,
                 # 2, 2, 2, 2, 2, 2, 2,
                 # 3, 3, 3, 3, 3, 3, 3,
                 # 4, 4, 4, 4, 4, 4, 4,
                 # 5, 5, 5, 5, 5, 5, 5,
                 # 6, 6, 6, 6, 6, 6, 6,
                 # 1, 1, 1, 1, 1, 1, 1,
                 # 7, 7, 7, 7, 7, 7, 7,
                 # 8, 8, 8, 8, 8, 8, 8,
                 # 9, 9, 9, 9, 9, 9, 9,
                 # 0, 0, 0, 0, 0, 0, 0,
                 # 0, 0, 0, 0, 0, 0, 0,
                 # 0, 0, 0, 0, 0, 0, 0,
                 # 0, 0, 0, 0, 0, 0, 0,
                 # 0, 0, 0, 0, 0, 0, 0,
                 # 0, 0, 0, 0, 0, 0, 0]

    # 1,0,1,1,1,0,0,1,0,1,0,0,1]
    # فایل برای ذخیره خروجی نهایی (Verification)
    all_audio = []

    for i, bit in enumerate(test_bits):
        for index in range(0, SAFETY_REPEAT):
           loop_start = time.time()
           frame = tx.generate_frame(bit)
           all_audio.append(frame)
           process_time = time.time() - loop_start
           if process_time < 0.02:
               time.sleep(0.02 - process_time)


    # پایان کار: ذخیره در فایل به جای پخش
    final_signal = np.concatenate(all_audio)
    wav.write("../tx_output_debug.wav", FS, final_signal)
    print("خروجی در فایل 'tx_output_debug.wav' ذخیره شد.")


if __name__ == "__main__":
    run_simulation()
