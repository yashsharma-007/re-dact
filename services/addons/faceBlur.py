import cv2
import numpy as np

def detect_and_blur_faces(input_path, output_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face_blurred = cv2.GaussianBlur(face, (99, 99), 30)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (w // 2, h // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        face_blurred = cv2.bitwise_and(face_blurred, face_blurred, mask=mask)
        mask_inv = cv2.bitwise_not(mask)
        face_original = cv2.bitwise_and(face, face, mask=mask_inv)
        
        face_final = cv2.add(face_original, face_blurred)
        image[y:y+h, x:x+w] = face_final
    
    cv2.imwrite(output_path, image)
    return output_path
