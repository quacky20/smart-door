import os
import cv2
import dlib
from embedding import *
from face_recognition import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_SRC_DIR = os.path.join(SCRIPT_DIR, "known_faces")
detector = dlib.get_frontal_face_detector()

if not os.path.exists(IMAGE_SRC_DIR):
    os.makedirs(IMAGE_SRC_DIR)
    print(f"Created empty directory at '{IMAGE_SRC_DIR}'. Please add subfolders with images there.")
    exit(0)

def process_static_image(img_path):
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"  [Warning] Could not load image: {img_path}")
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    if len(faces) == 0:
        print(f"  [Warning] No face detected in: {img_path}")
        return None
    
    face = max(faces, key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top()))

    h, w, _ = frame.shape
    x1, y1 = max(0, face.left()), max(0, face.top())
    x2, y2 = min(w, face.right()), min(h, face.bottom())
    cropped_face = frame[y1:y2, x1:x2]

    if cropped_face.size == 0:
        return None

    return get_face_embedding(cropped_face)

def run_ingestion_pipeline():
    print("Starting data ingestion pipeline...")
    total_added = 0

    for person_name in os.listdir(IMAGE_SRC_DIR):
        person_dir = os.path.join(IMAGE_SRC_DIR, person_name)
        
        if not os.path.isdir(person_dir):
            continue 
            
        print(f"\nProcessing identity group: {person_name}")
        
        for filename in os.listdir(person_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue
                
            img_path = os.path.join(person_dir, filename)
            print(f" -> Parsing {filename}...")
            
            embedding = process_static_image(img_path)
            if embedding is not None:
                register_face(person_name, embedding)
                total_added += 1

    print(f"\n[Finished] Successfully processed and added {total_added} profiles.")
    conn.close()

if __name__ == "__main__":
    run_ingestion_pipeline()
