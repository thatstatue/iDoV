import numpy as np
import sounddevice as sd
import queue
import sys

# --- تنظیمات ---
FS = 44100
BAUD_RATE = 10
BLOCK_DURATION = 1.0 / BAUD_RATE  # طول هر بلوک دقیقا اندازه یک بیت
BLOCK_SIZE = int(FS * BLOCK_DURATION)

FREQ_0 = 1200
FREQ_1 = 2200
THRESHOLD = 0.1  # آستانه انرژی برای تشخیص اینکه نویز نیست

# صفی برای انتقال دیتا از ترد صوتی به ترد اصلی (اختیاری، برای چاپ تمیز)
q = queue.Queue()


def process_block(indata, frames, time, status):
    """
    این تابع به صورت خودکار توسط sounddevice فراخوانی می‌شود.
    هر بار یک تکه (Block) از صدا را می‌گیرد.
    """
    if status:
        print(status, sys.stderr)

    # گرفتن دیتای مونو (تک کانال)
    audio_chunk = indata[:, 0]

    # 1. گرفتن تبدیل فوریه (FFT) روی همین تکه کوچک
    fft_vals = np.abs(np.fft.rfft(audio_chunk))
    freqs = np.fft.rfftfreq(len(audio_chunk), 1 / FS)

    # 2. پیدا کردن ایندکس فرکانس‌های هدف
    idx_0 = (np.abs(freqs - FREQ_0)).argmin()
    idx_1 = (np.abs(freqs - FREQ_1)).argmin()

    # 3. اندازه گیری انرژی در فرکانس‌های 1200 و 2200
    power_0 = fft_vals[idx_0]
    power_1 = fft_vals[idx_1]

    # 4. تشخیص بیت (Decision Logic)
    detected_bit = None

    # چک می‌کنیم آیا اصلا سیگنالی هست یا سکوت است؟
    if max(power_0, power_1) > THRESHOLD:
        if power_1 > power_0 * 1.5:  # اگر انرژی 2200 بیشتر بود
            detected_bit = 1
        elif power_0 > power_1 * 1.5:  # اگر انرژی 1200 بیشتر بود
            detected_bit = 0

    if detected_bit is not None:
        # نتیجه را در صف می‌گذاریم تا در حلقه اصلی چاپ شود
        q.put(detected_bit)
    else:
        # نویز یا سکوت
        pass


def main():
    print("--- FSK Receiver (Always Listening) ---")
    print(f"Listening for frequencies: {FREQ_0}Hz (0) and {FREQ_1}Hz (1)")
    print("Press Ctrl+C to stop.")

    # پیدا کردن دستگاه ورودی (اختیاری: می‌توان گذاشت روی default)
    # اما در لینوکس بهتر است با Pavucontrol ورودی را روی Monitor of Phone_To_Matlab بگذارید

    try:
        # باز کردن استریم صوتی
        with sd.InputStream(callback=process_block,
                            channels=1,
                            samplerate=FS,
                            blocksize=BLOCK_SIZE,
                            device=None):  # صریحاً روی None بگذارید
            while True:
                try:
                    # دریافت بیت‌های دیکد شده از تابع callback
                    bit = q.get(timeout=0.5)
                    print(f"Detected Bit: {bit}", True)
                except queue.Empty:
                    pass  # اگر دیتایی نیامد، ادامه بده

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()