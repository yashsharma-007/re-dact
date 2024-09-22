import os
from services.addons.faceBlur import detect_and_blur_faces
from services.addons.digiSignBlur import detect_and_blur_signatures  
from services.addons.digiSignBlur2 import detect_and_blur_signatures_doc  
from services.addons.removeMetadata import remove_metadata
from services.addons.numPlateBlur import detect_and_blur_license_plate
from services.addons.ocrBlur import redact_text_with_ocr

def process_image(file_path, redact_ocr, redact_meta, redact_face, redact_license_plate, redact_signature, redact_nsfw, is_document, sensitivity_level):
    
    if redact_meta:
        no_metadata_path = os.path.splitext(file_path)[0] + '_no_metadata' + os.path.splitext(file_path)[1]
        remove_metadata(file_path, no_metadata_path)
    else:
        no_metadata_path = file_path  

    
    if redact_face:
        faces_blurred_path = os.path.splitext(no_metadata_path)[0] + '_faces_blurred' + os.path.splitext(no_metadata_path)[1]
        detect_and_blur_faces(no_metadata_path, faces_blurred_path)
    else:
        faces_blurred_path = no_metadata_path  

    
    if redact_signature:
        if is_document:
            signatures_blurred_path = os.path.splitext(faces_blurred_path)[0] + '_signatures_blurred' + os.path.splitext(faces_blurred_path)[1]
            detect_and_blur_signatures_doc(faces_blurred_path, signatures_blurred_path)  
        else:
            signatures_blurred_path = os.path.splitext(faces_blurred_path)[0] + '_signatures_blurred' + os.path.splitext(faces_blurred_path)[1]
            detect_and_blur_signatures(faces_blurred_path, signatures_blurred_path)  
    else:
        signatures_blurred_path = faces_blurred_path  

    
    if redact_license_plate and not is_document:
        final_output_path = os.path.splitext(signatures_blurred_path)[0] + '_plates_blurred' + os.path.splitext(signatures_blurred_path)[1]
        detect_and_blur_license_plate(signatures_blurred_path, final_output_path)
    else:
        final_output_path = signatures_blurred_path  

    
    if redact_ocr:
        ocr_blurred_path = os.path.splitext(final_output_path)[0] + '_ocr_blurred' + os.path.splitext(final_output_path)[1]
        redact_text_with_ocr(final_output_path, ocr_blurred_path)  
        final_output_path = ocr_blurred_path

    return final_output_path
