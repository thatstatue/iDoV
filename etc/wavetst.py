import numpy as np
import sounddevice as sd
import time
import random

# --- تنظیمات ---
FS = 44100  # نرخ نمونه‌برداری
BAUD_RATE = 10  # نرخ ارسال (تعداد بیت در ثانیه) - برای شروع کم می‌گذاریم
FREQ_0 = 1200  # فرکانس بیت 0
FREQ_1 = 2200  # فرکانس بیت 1
AMPLITUDE = 0.5  # بلندی صدا


def generate_fsk_signal(bits):
    """رشته بیت‌ها را می‌گیرد و سیگنال صوتی تولید می‌کند"""
    duration_per_bit = 1.0 / BAUD_RATE
    t = np.linspace(0, duration_per_bit, int(FS * duration_per_bit), endpoint=False)

    signal = np.array([])

    # اضافه کردن یک هدر (Preamble) برای اینکه گیرنده بفهمد پیام شروع شده
    # مثلا 0.5 ثانیه فرکانس 500 هرتز
    header_t = np.linspace(0, 0.5, int(FS * 0.5), endpoint=False)
    header = AMPLITUDE * np.sin(2 * np.pi * 500 * header_t)
    signal = np.concatenate((signal, header))

    for bit in bits:
        if bit == 0:
            wave = AMPLITUDE * np.sin(2 * np.pi * FREQ_0 * t)
        else:
            wave = AMPLITUDE * np.sin(2 * np.pi * FREQ_1 * t)

        # برای جلوگیری از تیک زدن (Phase Continuity) بهتر است فاز را محاسبه کنیم
        # اما برای سادگی در این مرحله صرف نظر می‌کنیم
        signal = np.concatenate((signal, wave))

    return signal


def main():
    print("--- FSK Sender ---")

    # 1. تولید دیتای رندوم (مثلا 8 بیت)
    random_bits = [random.choice([0, 1]) for _ in range(8)]
    print(f"Sending Bits: {random_bits}")

    # 2. مدولاسیون (ساخت سیگنال)
    audio_signal = generate_fsk_signal(random_bits)

    # 3. پخش صدا
    # نکته: در لینوکس بعد از اجرا، باید با Pavucontrol خروجی این اسکریپت
    # را روی "Matlab_To_Phone" تنظیم کنید.
    print("Playing audio... (Check PulseAudio settings)")
    sd.play(audio_signal, samplerate=FS)
    sd.wait()  # منتظر ماندن تا پخش تمام شود
    print("Done.")


if __name__ == "__main__":
    main()