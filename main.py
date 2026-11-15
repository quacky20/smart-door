import time
import cv2
import dlib
from embedding import get_face_embedding
from face_recognition import recognize_face
from face_utils import scale_rect, largest_face, extract_face
from variables import *

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

window_name = 'Video Stream'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 960, 600)

detector = dlib.get_frontal_face_detector()


def unlock_door():
    # TODO: Implement door unlock function (GPIO / relay)
    print("[UNLOCK] Door unlocked")


current_name = "Unknown"
streak_name = None            # identity currently being confirmed
streak = 0                    # consecutive recognitions of streak_name
frames_since_recog = RECOGNIZE_EVERY_N   # forces recognition on first face frame
last_unlock = 0.0

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    small_frame = cv2.resize(frame, (0, 0), fx=DETECT_RATIO, fy=DETECT_RATIO)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, DETECT_UPSAMPLE)

    if len(faces) == 0:
        # Face gone: reset everything so a new person must be confirmed from scratch.
        current_name, streak_name, streak = "Unknown", None, 0
        frames_since_recog = RECOGNIZE_EVERY_N
    else:
        # Door lock: only the largest (closest) face is considered.
        face = scale_rect(largest_face(faces), DETECT_RATIO)
        frames_since_recog += 1

        if frames_since_recog >= RECOGNIZE_EVERY_N:
            frames_since_recog = 0
            face_img = extract_face(frame, face)

            if face_img is not None:
                name, distance = recognize_face(get_face_embedding(face_img))
                current_name = name

                if name == "Unknown":
                    streak_name, streak = None, 0
                elif name == streak_name:
                    streak += 1
                else:
                    streak_name, streak = name, 1

                if streak >= CONFIRM_RECOGNITIONS and time.time() - last_unlock > UNLOCK_COOLDOWN_S:
                    unlock_door()
                    last_unlock = time.time()

        h, w, _ = frame.shape
        x1, y1 = max(0, face.left()), max(0, face.top())
        x2, y2 = min(w, face.right()), min(h, face.bottom())

        confirmed = streak >= CONFIRM_RECOGNITIONS
        if current_name == "Unknown":
            colour, label = (0, 0, 255), "Unknown"
        elif confirmed:
            colour, label = (0, 255, 0), current_name
        else:
            colour, label = (0, 255, 255), f"{current_name}?"

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)

    cv2.imshow(window_name, frame)

    keyPressed = cv2.waitKey(1) & 0xFF

    if keyPressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
