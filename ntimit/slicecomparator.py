
import os
import soundfile as sf

from ntimit.frcodec import run_vocoder_simulation
from ntimit.utilities import extract_voice_features, load_wav, extract_bandwidth, extract_frame_count, frame_signal


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




def slice_wav_by_frames(input_wav, output_wav, start_frame, end_frame, frame_ms=20):
    data, sr = sf.read(input_wav)
    if data.ndim > 1:
        data = data[:, 0]  # mono
    samples_per_frame = int(sr * frame_ms / 1000)
    start_sample = start_frame * samples_per_frame
    end_sample = end_frame * samples_per_frame
    sliced = data[start_sample:end_sample]
    sf.write(output_wav, sliced, sr)




def comparator(ogpath, decpath):

    fs, original = load_wav(ogpath)
    bandwidth_og = extract_bandwidth(original, fs)
    print(ogpath , " frames: ", len(frame_signal(original, fs)))

    fs, decoded = load_wav(decpath)
    bandwidth_dec = extract_bandwidth(decoded, fs)
    print(decoded , " frames: ", len(frame_signal(original, fs)))

    t = 0
    for i in range(len(bandwidth_og)):
        if bandwidth_og[i] != bandwidth_dec[i]:
           # print("diff in ", ogpath, i)
            t = t +1
    print( "count of diffs in ", ogpath, "and" , decpath, " are ", t)

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
        inp = name + str(start) +".wav"
        out = name+ str(start) +"_FR.wav"
        comparator(inp, out)
        start =start+l

def test_slices_WAV(name,start,finish,l):

    while start+l<=finish:
        inp = name+ str(start) +".WAV"
        out = name+ str(start) +"_FR.WAV"
        comparator(inp, out)
        start =start+l


def vocode_slices_WAV(name,start,finish,l):

    while start+l<=finish:  # ran/frame_count
        inp = name + str(start) +".WAV"
        out = name + str(start) +"_FR.WAV"
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

def slice_and_test_WAV(og_full, test_half, start, finish, l):
    #l = 12
    #testhalf = "test/SA1-3_"
    s = start
    while start + l <= finish:  # ran/frame_count
        slice_up_WAV(og_full, test_half, start, l)
        start = start + l

    test_winners_WAV(test_half, s, finish, l)
    # comparator("test/SA1-3_115.wav","recorded/REC_SA1-3_115.wav")

def test_the_alphabet():
    comparator("test/SA1-3_115.WAV", "test/SA1-3_115_FR.WAV")
    comparator("test/SA1-3_127.WAV", "test/SA1-3_127_FR.WAV")
    comparator("test/SA1-3_139.WAV", "test/SA1-3_139_FR.WAV")
    comparator("test/SA1-3_151.WAV", "test/SA1-3_151_FR.WAV")

    print(os.path.exists("test/SA2_215.wav"))
    path = "test/SA2_215.wav"
    comparator("test/SA2_215.wav", "test/SA2_215_FR.WAV")

    # comparator("test/SA2-2_97.WAV", "test/SA2-2_97_FR.WAV")
    # comparator("test/SA2-2_124.WAV", "test/SA2-2_124_FR.WAV")
    # comparator("test/SA2-2_132.WAV", "test/SA2-2_132_FR.WAV")


    # comparator("test/SA2_148.WAV", "test/SA2_148_FR.WAV")
    # comparator("test/SA2_160.WAV", "test/SA2_160_FR.WAV")

    # comparator("test/SA2_188.WAV", "test/SA2_188_FR.WAV")
    # comparator("test/SA2_200.WAV", "test/SA2_200_FR.WAV")
#
# 50.0 11 28 18
# 50.0 35 50 16
# -50.0 60 75 16
# 50.0 88 101 14
# 50.0 146 122 11
# 50.0 134 157 24
def add_to_alphabet():
    name ="OG/SI648.WAV"
    enc = "OG/SI648_ENC.wav"
    #run_vocoder_simulation(name,  enc, "exe/fr_enc.exe", "exe/fr_dec.exe")
    #longest_no_distortion(name,enc)
    s=146
    slice_and_test_WAV(name ,"test/SI648_",s,s+12,12)

if __name__ == "__main__":
    comparator("OG/SX217.WAV","test_results/5991327444211853234_encrypted_AMR.wav")
    #add_to_alphabet()

