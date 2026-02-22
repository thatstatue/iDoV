#todo: ایده اینه که یکی درمیون آپرکیس لوورکیس باشن که اگه چند بار دیتا ریپیت شد به مشکل نخوریم

from ntimit.virtualmic import play_wav

voices = ["test/SI648_146.WAV",     #0
          "test/SI648_134.WAV",     #1
          "test/SI648_88.WAV",      #2
          "test/SI648_60.WAV",      #3
          "test/SI648_11.WAV",      #4
          "test/SA1-3_127.WAV",     #5    #8
          "test/SA1-3_151.WAV",     #9
          "test/SA2_84.WAV",        #a
          "test/SA2_100.WAV",       #b
          "test/SA2_119.WAV",       #c
          "test/SA2_148.WAV",       #d
          "test/SA2_160.WAV",
          "test/SA2-2_97.WAV",      #16
          "test/SA2-2_124.WAV",
          "test/SA2-2_132.WAV",
          "test/SA1-3_115.WAV",
          "test/SI648_35.WAV",
          "test/SA1-3_139.WAV",
          ]

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
        play_wav(voices[path[j]], device=device)

if __name__ == "__main__":
    #play_wav("OG/SA2.WAV", device=device) for changing inp dev
    inp = input("write something:\n")
    enc = encrypt(inp)
    play_to_mic(enc)
