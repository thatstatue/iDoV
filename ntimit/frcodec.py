
import subprocess
import os
import time


def run_vocoder_simulation(input_wav_path, output_wav_path, encoder, decoder):
    # 1. تبدیل فایل WAV به فرمت خام (RAW) که معمولا این exeها نیاز دارند
    raw_input = "temp_input.raw"
    encoded_file = "TEMP.bit"
    decoded_raw = "temp_input.raw"

    # تبدیل wav به raw (فرکانس 8000 هرتز معمولا برای این کدک‌ها استاندارد است)
    subprocess.call(['ffmpeg', '-i', input_wav_path, '-f', 's16le', '-ar', '8000', '-ac', '1', raw_input, '-y'])

    # فرض: ورودی فایل raw و خروجی فایل فشرده است
    print("Running Encoder...")
    subprocess.call(['wine', encoder, raw_input, encoded_file])
    print("Running Decoder...")
    subprocess.call(['wine', decoder, encoded_file, decoded_raw])

    # 4. تبدیل raw برگشتی به wav برای پردازش نهایی
    subprocess.call(['ffmpeg', '-f', 's16le', '-ar', '8000', '-ac', '1', '-i', decoded_raw, output_wav_path, '-y'])

    # پاک کردن فایل‌های موقت
#    os.remove(raw_input)

    return output_wav_path


if __name__ == "__main__":
    run_vocoder_simulation("OG/SA1-3.WAV", "SA1-3_ENC_FR.wav", '../exe/fr_enc.exe', '../exe/fr_dec.exe')
