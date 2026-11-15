# Smart Door - Face Recognition Lock
## Working
---
## Requirements
---
## Setup
 - [Download FaceNet model](https://github.com/pillarpond/face-recognizer-android/blob/master/app/src/main/assets/facenet.tflite)
 - Download [dlib's 5-point landmark model](http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2), extract it and place `shape_predictor_5_face_landmarks.dat` next to `main.py` (enables face alignment)
 - Install requirements from `requirements.txt`
 - Put enrollment photos in `known_faces/<person_name>/` and run `python ingest_data.py`
 - **If you change alignment on/off or the model, delete `face_data.db` and `face_index.faiss` and re-run ingestion** - old embeddings are not comparable.
 - Tune thresholds in `variables.py` (`MATCH_THRESHOLD_L2`, `DETECT_RATIO`, `RECOGNIZE_EVERY_N`, `CONFIRM_RECOGNITIONS`)
---
## References
 - [Raspberry Pi-Based Face Recognition Door Lock System (Seifeldin Sherif Fathy Ali Elnozahy, Senthill C. Pari * and Lee Chu Liang)](https://www.mdpi.com/2624-831X/6/2/31)