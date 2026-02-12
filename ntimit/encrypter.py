from ntimit.virtualmic import play_wav
voices_start = 16
voices_finish = 17
device = 'pulse'
voices = ["test/SI648_146.WAV",
          "test/SI648_134.WAV",
          "test/SI648_88.WAV",
          "test/SI648_60.WAV",
          "test/SI648_11.WAV",
          "test/SA1-3_115.WAV",
          "test/SA1-3_127.WAV",
          "test/SA2-2_124.WAV",
          "test/SA1-3_139.WAV",
          "test/SA1-3_151.WAV",
          "test/SA2_84.WAV",
          "test/SA2_100.WAV",
          "test/SA2_119.WAV",
          "test/SA2_148.WAV",
          "test/SA2_160.WAV",
          "test/SA2_188.WAV",
          #"test/SA2_200.WAV",
          "test/SA2-2_97.WAV",
          "test/SI648_35.WAV",
          "test/SA2-2_132.WAV",
          ]
def text_to_hex(text):
    return text.encode('utf-8').hex()


def hex_to_voice(hex):
    i = 0
    j = 0
    path = []
    while i < len(str(hex)):
        n = hex[i]
        if 'a' <= hex[i] <= 'f':
            n = ord(n) - ord('a') + 10
        path.append(int(n))
        i+=1
    return path

def encrypt(text):
    return hex_to_voice(text_to_hex(text))


def play_to_mic(path):
    play_wav(voices[voices_start], device=device)
    for j in range(len(path)):
        play_wav(voices[path[j]], device=device)
    play_wav(voices[voices_finish], device=device)


if __name__ == "__main__":
    inp = input("write something!")
    enc = encrypt(inp)
    play_to_mic(enc)
