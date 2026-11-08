import ai_edge_litert.interpreter as litert
import cv2
import numpy as np

interpreter = litert.Interpreter(model_path="facenet.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

def get_face_embedding(cropped_face):
    face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (input_shape[1], input_shape[2]))
    
    face_normalized = (face_resized.astype(np.float32) - 127.5) / 128.0
    input_data = np.expand_dims(face_normalized, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    embedding = interpreter.get_tensor(output_details[0]['index'])[0]
    
    embedding = embedding / np.linalg.norm(embedding)
    return embedding