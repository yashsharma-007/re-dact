import cv2
import numpy as np
from skimage import measure, morphology
from skimage.measure import regionprops


constant_parameter_1 = 84
constant_parameter_2 = 250
constant_parameter_3 = 100
constant_parameter_4 = 18

def detect_and_blur_signatures_doc(input_path, output_path):
    
    image = cv2.imread(input_path)

    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 9)

    
    blobs_labels = measure.label(binary, background=0)

    
    total_area = 0
    counter = 0
    the_biggest_component = 0

    for region in regionprops(blobs_labels):
        if region.area > 10:
            total_area += region.area
            counter += 1
            if region.area >= constant_parameter_2 and region.area > the_biggest_component:
                the_biggest_component = region.area

    average = total_area / counter if counter > 0 else 0

    
    a4_small_size_outlier_constant = ((average / constant_parameter_1) * constant_parameter_2) + constant_parameter_3
    a4_big_size_outlier_constant = a4_small_size_outlier_constant * constant_parameter_4

    
    filtered_blobs = morphology.remove_small_objects(blobs_labels, a4_small_size_outlier_constant)
    component_sizes = np.bincount(filtered_blobs.ravel())
    too_big_mask = component_sizes > a4_big_size_outlier_constant
    too_big_mask = too_big_mask[filtered_blobs]
    filtered_blobs[too_big_mask] = 0

    
    for region in regionprops(filtered_blobs):
        if a4_small_size_outlier_constant <= region.area <= a4_big_size_outlier_constant:
            
            min_row, min_col, max_row, max_col = region.bbox
            
            signature_roi = image[min_row:max_row, min_col:max_col]
            
            blurred_signature = cv2.GaussianBlur(signature_roi, (51, 51), 0)
            
            image[min_row:max_row, min_col:max_col] = blurred_signature

    
    cv2.imwrite(output_path, image)
    return output_path
