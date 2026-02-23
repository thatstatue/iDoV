
buffer = "recorded/buffer.WAV"
INPUT_DEVICE = 'pulse'
OUTPUT_DEVICE = 'pulse'

SAMPLE_RATE = 16000
BLOCK_SIZE_SECONDS = 0.24
SILENCE_THRESHOLD = 8
VOICE_SIGNATURES = []
SIMILARITY_THRESHOLD = 0
OUTPUT_DIR = "recorded"
SEARCH_RADIUS = 120  # samples (~7.5ms)
SEARCH_STEP = 5  # resolution
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_SIZE_SECONDS)  # 3840
DEBOUNCE_SECONDS = 0.20  # ignore detections within this time window

پBLOCK_DURATION = 0.24
STEP_SIZE = 80  # 5ms shift
DETECTION_THRESHOLD = 0.80  # strong match only

HEX_DATA = []

VOICES = ["test/SI648_146.WAV",
          "test/SI648_134.WAV",
          "test/SI648_88.WAV",
          "test/SI648_60.WAV",
          "test/SI648_11.WAV",
          "test/SA1-3_127.WAV",
          "test/SA1-3_151.WAV",
          "test/SA2_84.WAV",
          "test/SA2_100.WAV",
          "test/SA2_119.WAV",
          "test/SA2_148.WAV",
          "test/SA2_160.WAV",
          "test/SA2-2_97.WAV",
          "test/SA2-2_124.WAV",
          "test/SA2-2_132.WAV",
          "test/SA1-3_115.WAV",
          "test/SI648_35.WAV",
          "test/SA1-3_139.WAV",
          ]