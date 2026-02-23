
buffer = "recorded/buffer.WAV"
device = 'pulse'
SAMPLE_RATE = 16000
BLOCK_SIZE_SECONDS = 0.24
#HEX_DATA = []
CHAR_DATA = []
SILENCE_THRESHOLD = 1
VOICE_SIGNATURES = []
SIMILARITY_THRESHOLD = 0

SEARCH_RADIUS = 120  # samples (~7.5ms)
SEARCH_STEP = 10  # resolution
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_SIZE_SECONDS)  # 3840
DEBOUNCE_SECONDS = 0.20  # ignore detections within this time window

SAMPLE_RATE = 16000
BLOCK_DURATION = 0.24
STEP_SIZE = 80  # 5ms shift
DETECTION_THRESHOLD = 0.80  # strong match only

HEX_DATA = []

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