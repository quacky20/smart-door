DB_FILE = "face_data.db"
INDEX_FILE = "face_index.faiss"
EMBEDDING_DIM = 512

# --- Recognition ---
# Match threshold as a real (non-squared) L2 distance between unit-normalised
# embeddings. Default ~= sqrt(0.6), i.e. the same effective cut-off the old code
# had. Calibrate on your own genuine/impostor pairs before trusting it.
MATCH_THRESHOLD_L2 = 0.7746

# --- Alignment ---
# dlib 5-point landmark model (see README). If the file is missing the code
# falls back to a plain padded crop.
LANDMARK_MODEL = "shape_predictor_5_face_landmarks.dat"
FACE_SIZE = 160          # chip size fed to FaceNet
FACE_PADDING = 0.25

# --- Live loop (main.py) ---
DETECT_RATIO = 0.5       # frame downscale before HOG detection
DETECT_UPSAMPLE = 0      # 0 = fastest; 1 finds smaller faces but ~4x slower
                         # NOTE: dlib HOG needs ~80px faces *after* downscaling
RECOGNIZE_EVERY_N = 3    # run FaceNet only every N frames while a face is present
CONFIRM_RECOGNITIONS = 3 # consecutive same-identity matches needed to unlock
UNLOCK_COOLDOWN_S = 5.0  # min seconds between unlock triggers
