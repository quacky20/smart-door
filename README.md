# Smart Door - Face Recognition Lock
 
A face recognition door access prototype that runs on a laptop webcam. The door-unlock step is currently a placeholder that prints a message; Raspberry Pi and relay/lock integration is planned.
 
## Working
 
Dlib finds the face in each webcam frame, and a FaceNet (TFLite) model turns it into a 512-d embedding. FAISS finds the closest enrolled embedding, and SQLite maps that match back to a name. If the distance is under the threshold for several consecutive checks, the door "unlocks" (with a cooldown between unlocks).
 
## Requirements
 
- Python 3
- OpenCV, dlib, LiteRT, FAISS, NumPy (see `requirements.txt`)
- A webcam
## Setup
 
 - [Download FaceNet model](https://github.com/pillarpond/face-recognizer-android/blob/master/app/src/main/assets/facenet.tflite) and place `facenet.tflite` next to `main.py`
 - Download [dlib's 5-point landmark model](http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2), extract it and place `shape_predictor_5_face_landmarks.dat` next to `main.py` (enables face alignment)
 - Install requirements from `requirements.txt`
 - Put enrollment photos in `known_faces/<person_name>/` (png, jpg, jpeg or webp) and run `python ingest_data.py`
 - Run `python main.py`
 - **If you change alignment on/off or the model, delete `face_data.db` and `face_index.faiss` and re-run ingestion** - old embeddings are not comparable.
 - Tune thresholds in `variables.py` (`MATCH_THRESHOLD_L2`, `DETECT_RATIO`, `RECOGNIZE_EVERY_N`, `CONFIRM_RECOGNITIONS`)
---
## References
 - S. S. F. A. Elnozahy, S. C. Pari, and L. C. Liang, [Raspberry Pi-Based Face Recognition Door Lock System](https://www.mdpi.com/2624-831X/6/2/31), *IoT*, vol. 6, no. 2, 2025.
 - F. Schroff, D. Kalenichenko, and J. Philbin, FaceNet: A Unified Embedding for Face Recognition and Clustering, CVPR 2015.
 - [Dlib](https://dlib.net/), [FAISS](https://github.com/facebookresearch/faiss), [OpenCV](https://docs.opencv.org/)
 - [Face Recognition Dataset (Kaggle)](https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset), used for evaluation
 