#todo: ایده اینه که یکی درمیون آپرکیس لوورکیس باشن که اگه چند بار دیتا ریپیت شد به مشکل نخوریم
from ntimit.slicecomparator import comparator
from ntimit.utilities import decrypt_voice, load_voice_signatures
from consts import VOICES, VOICE_SIGNATURES
from ntimit.frcodec import run_vocoder_simulation
from ntimit.utilities import concatenate_wav_files_wave
from ntimit.virtualmic import play_wav


voices_start = 16
voices_finish = 17
device = 'pulse'
def text_to_hex(text):
    v = text.encode('utf-8').hex()
    print("hex: ",v)
    return v


def hex_to_voice(hex):
    i = 0
    j = 0
    path = [voices_start,voices_start]
    while i < len(str(hex)):
        n = hex[i]
        if 'a' <= hex[i] <= 'f':
            n = ord(n) - ord('a') + 10
        path.append(int(n))
        i+=1
    path.append(voices_finish)
    path.append(voices_finish)
    print(path)
    return path

def encrypt(text):
    return hex_to_voice(text_to_hex(text))

def play_to_mic(path):
    for j in range(len(path)):
        play_wav(VOICES[path[j]], device=device)
def hex_to_array(hex):
    a = [16,16]
    b = str(hex)
    for i in range(len(b)) :
         a.append(int (hex[i], 16))
    a.append(17)
    a.append(17)
    return a
def main_for_phoning():
    #play_wav("OG/SA2.WAV", device=device) for changing inp dev
    inp = input("write something:\n")
    enc = encrypt(inp)
    play_to_mic(enc)


def run_all_vocoders(outp , pre, h):
    codedE = pre + h + "_encrypted_EFR.wav"
    codedER = pre + h + "_encrypted_EFR_R.wav"

    codedF = pre + h + "_encrypted_FR.wav"
    codedFR = pre + h + "_encrypted_FR_R.wav"

    codedA = pre + h + "_encrypted_AMR.wav"
    codedAR = pre + h + "_encrypted_AMR_TW.wav"
    codedAT = pre + h + "_encrypted_AMR_TSEQ.wav"

    run_vocoder_simulation(outp, codedE, '../exe/gsmefr-encode.exe', '../exe/gsmefr-decode.exe')
    comparator(outp, codedE)

    # run_vocoder_simulation(outp, codedER, '../exe/gsmefr-encode-r.exe', '../exe/gsmefr-decode-r.exe')
    # comparator(outp, codedER)
    #
    run_vocoder_simulation(outp,codedF, '../exe/gsmfr-encode.exe', '../exe/gsmfr-decode.exe')
    comparator(outp,codedF)
    # run_vocoder_simulation(outp, codedFR, '../exe/gsmfr-encode-r.exe', '../exe/gsmfr-decode-r.exe')
    # comparator(outp, codedFR)

    run_vocoder_simulation(outp, codedA, '../exe/amr_encoder.exe.exe', '../exe/amr_decoder.exe.exe')
    comparator(outp, codedA)
    run_vocoder_simulation(outp, codedAR, '../exe/twamr-encode.exe', '../exe/atwamr-decode.exe')
    comparator(outp, codedAR)
    run_vocoder_simulation(outp, codedAT, '../exe/twamr-tseq-enc.exe', '../exe/twamr-tseq-dec.exe')
    comparator(outp, codedAT)
    return codedE, codedF, codedA,codedAR, codedAT


if __name__ == "__main__":
    load_voice_signatures()
    print(f"Loaded {len([v for v in VOICE_SIGNATURES if v is not None])} voice signatures")

    inp = input("write something:\n")
    a = hex_to_array( text_to_hex(inp))
    h =str(hash(inp))
    pre = "test_results/"
    outp= pre+h+"_encrypted.wav"
    concatenate_wav_files_wave(a,outp)
    e,f,a , ar,at= run_all_vocoders(outp, pre, h)
    decrypt_voice(outp)
    decrypt_voice(e)
    decrypt_voice(at)
    decrypt_voice(ar)



