import cv2
import os
from ultralytics import YOLO

def detect_and_blur_signatures(input_path, output_path, model_path=os.path.join('services', 'addons', 'best.pt'), conf_threshold=0.75):
    model = YOLO(model_path)
    
    image = cv2.imread(input_path)
    
    results = model.predict(source=input_path, conf=conf_threshold)

    print(f"Number of detections: {len(results)}")

    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            
            print(f"Number of boxes detected: {len(boxes)}")
            
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                
                print(f"Box {idx + 1}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                
                signature_roi = image[y1:y2, x1:x2]

                blurred_signature = cv2.GaussianBlur(signature_roi, (51, 51), 0)

                image[y1:y2, x1:x2] = blurred_signature

    cv2.imwrite(output_path, image)

    return output_path
