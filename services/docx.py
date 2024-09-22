from docx import Document
from docx.oxml.ns import qn
from docx.shared import Inches
import os
import tempfile

def redact_paragraphs(paragraphs, sensitivity_level):
    """
    Redact text in the paragraphs of a DOCX document.
    """
    from services.txt import redact_text

    for para in paragraphs:
        if para.text:
            redacted_text = redact_text(para.text, sensitivity_level)
            para.text = redacted_text
    return paragraphs

def process_images_and_text(doc, upload_folder, sensitivity_level, redact_ocr, redact_meta, redact_face, redact_license_plate, redact_signature, redact_nsfw, is_document):
    """
    Traverse through the document paragraphs and process both text and images.
    """
    from services.img import process_image
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            if run._element.xpath('.//a:blip'):
                blip = run._element.xpath('.//a:blip')[0]
                embed_rid = blip.get(qn('r:embed'))
                image_part = doc.part.related_parts[embed_rid]
                image_filename = os.path.join(upload_folder, f"temp_image_{embed_rid}.png")
                with open(image_filename, 'wb') as img_file:
                    img_file.write(image_part.blob)
                processed_image = process_image(image_filename, redact_ocr, redact_meta, redact_face,
                                                redact_license_plate, redact_signature, redact_nsfw, is_document, 
                                                sensitivity_level)

                with open(processed_image, 'rb') as img_file:
                    image_part._blob = img_file.read()

            if run.text:
                from services.txt import redact_text
                redacted_text = redact_text(run.text, sensitivity_level)
                run.text = redacted_text

def process_docx_file(docx_file, sensitivity_level, redact_ocr, redact_meta, redact_face, redact_license_plate, redact_signature, redact_nsfw, is_document):
    """
    Process a DOCX file by redacting text and replacing processed images.
    """
    doc = Document(docx_file)
    upload_folder = 'static/uploads'

    process_images_and_text(doc, upload_folder, sensitivity_level, redact_ocr, redact_meta, redact_face, 
                            redact_license_plate, redact_signature, redact_nsfw, is_document)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.docx')
    doc.save(temp_file.name)

    return temp_file.name
