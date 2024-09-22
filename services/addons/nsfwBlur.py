import tensorflow as tf
import cv2
import numpy as np


model = tf.keras.models.load_model('nsfw.299x299.h5')

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the input image for NSFWJS model.

    Parameters:
        image (np.array): Input image.
        target_size (tuple): Target size for resizing the image.

    Returns:
        preprocessed_img (np.array): Image resized and normalized.
    """
    resized_image = cv2.resize(image, target_size)
    normalized_image = resized_image.astype('float32') / 255.0
    return np.expand_dims(normalized_image, axis=0)

def detect_and_blur_nsfw_js(input_path, output_path, threshold=0.85):
    """
    Detect and blur images using NSFWJS if they contain explicit content.

    Parameters:
        input_path (str): Path to the input image.
        output_path (str): Path to save the output blurred image.
        threshold (float): Threshold for detecting explicit content.

    Returns:
        output_path (str): Path to the processed image.
    """
    
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Unable to load image from {input_path}")

    preprocessed_image = preprocess_image(image)
    
    
    predictions = model.predict(preprocessed_image)
    
    
    class_names = ['neutral', 'drawing', 'hentai', 'porn', 'sexy']
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    
    if predicted_class in ['porn', 'hentai'] and confidence > threshold:
        blurred_image = cv2.GaussianBlur(image, (99, 99), 30)
        cv2.imwrite(output_path, blurred_image)
        print(f"NSFW content detected and blurred. Saved to {output_path}")
    elif predicted_class == 'sexy' and confidence > 0.7:
        blurred_image = cv2.GaussianBlur(image, (99, 99), 30)
        cv2.imwrite(output_path, blurred_image)
        print(f"Partial nudity detected and blurred. Saved to {output_path}")
    else:
        
        cv2.imwrite(output_path, image)
        print(f"Image is safe, saved without modification to {output_path}")

    return output_path
