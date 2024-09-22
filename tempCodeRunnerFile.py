from flask import Flask, render_template, request, send_file, redirect, url_for
from services.txt import redact_text
from services.img import process_image
from services.docx import process_docx_file
from services.pdf import process_pdf_file
from services.others import process_other_file
import os
import tempfile
import logging
from werkzeug.utils import secure_filename


logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

def determine_sensitivity_level(slider_value):
    """
    Determine the sensitivity level based on the slider input value.
    """
    if 0 <= slider_value <= 33:
        return 1
    elif 34 <= slider_value <= 66:
        return 2
    elif 67 <= slider_value <= 100:
        return 3
    else:
        return 1  

@app.route('/')
def index():
    """
    Render the index page where users can upload files.
    """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file uploads, process text or image files, and return results based on checkboxes.
    """
    if 'file' not in request.files:
        logging.error('No file part in the request')
        return 'No file part in the request'
    
    file = request.files['file']
    if file.filename == '':
        logging.error('No file selected')
        return 'No file selected'

    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1].lower()
    logging.info(f'Uploaded file: {filename}, Extension: {file_ext}')

    try:
        
        sensitivity = int(request.form.get('sensitivity', 1))
        sensitivity_level = determine_sensitivity_level(sensitivity)
        logging.info(f'Sensitivity Level: {sensitivity_level}')

        
        redact_ocr = 'redact_ocr' in request.form
        redact_meta = 'redact_meta' in request.form
        redact_face = 'redact_face' in request.form
        redact_license_plate = 'redact_license_plate' in request.form
        redact_signature = 'redact_signature' in request.form
        redact_nsfw = 'redact_nsfw' in request.form
        is_document = 'doc_check' in request.form

        
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        logging.info(f'File saved to {upload_path}')

        
        if file_ext == '.txt':
            with open(upload_path, 'r', encoding='utf-8') as f:
                text = f.read()
            logging.info('Processing text content for redaction...')
            redacted_text = redact_text(text, sensitivity_level)
            logging.info('Redaction complete')

            return render_template('result.html', redacted_text=redacted_text, original_text=text, file_path=upload_path, original_file=upload_path)

        
        
        elif file_ext == '.docx':
            logging.info('Processing DOCX file for redaction...')
            redacted_docx_path = process_docx_file(
                file, 
                sensitivity_level,
                redact_ocr,
                redact_meta,
                redact_face,
                redact_license_plate,
                redact_signature,
                redact_nsfw,
                is_document
            )
            logging.info(f'Redacted DOCX file saved at: {redacted_docx_path}')
            return render_template('result.html', docx_file_path=redacted_docx_path, original_file=upload_path)

        
        
        
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        return f"Error processing file: {str(e)}"

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """
    Allow users to download the processed file.
    """
    file_path = os.path.join(app.root_path, filename)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
