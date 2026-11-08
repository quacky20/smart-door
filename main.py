import cv2
import dlib
from embedding import get_face_embedding
from face_recognition import recognize_face

cap = cv2.VideoCapture(0)

window_name = 'Video Stream'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 960, 600)

detector = dlib.get_frontal_face_detector()

while True:
    ret, frame = cap.read()
    
    if ret == False:
        continue
    
    ratio = 0.4
    
    small_frame = cv2.resize(frame, (0, 0), fx=ratio, fy=ratio)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray, 1)
    
    for face in faces:
        x1 = int(face.left() / ratio)
        y1 = int(face.top() / ratio)
        x2 = int(face.right() / ratio)
        y2 = int(face.bottom() / ratio)
        
    
        h, w, _ = frame.shape
        x1, x2, y1, y2 = max(0, x1), min(w, x2), max(0, y1), min(h, y2)
        cropped_frame = frame[y1:y2, x1:x2]
        
        if cropped_frame.size > 0:
            embeddings = get_face_embedding(cropped_frame)
            
        name, distance = recognize_face(embeddings)
        # DEBUG: 
        # print(name)
        
        lock_open = name!="Unknown"
        # DEBUG: 
        # print(lock_open)
        
        colour = (0, 255, 0) if name!="Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
        
        # if lock_open: 
            # TODO: Implement door unlock function
    
    cv2.imshow(window_name, frame)
    
    keyPressed = cv2.waitKey(1) & 0xFF
    
    if keyPressed == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()