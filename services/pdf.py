import fitz  # PyMuPDF
import tempfile
import os
from services.txt import redact_text
from services.img import process_image

def process_pdf_file(file, sensitivity_level):
    """
    Process a PDF file: redact text, process images, and return the path to the redacted file.
    """
    # Save the uploaded PDF file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_input_file:
        file.save(temp_input_file.name)  # Save the uploaded file to the temporary file path

    doc = None  # Initialize doc to None to avoid the uninitialized variable issue

    try:
        # Open the saved PDF with fitz (PyMuPDF)
        doc = fitz.Document(temp_input_file.name)  # Use fitz.Document() instead of fitz.open()
        redacted_pdf = fitz.Document()  # Create new PDF

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")

            # Redact text using txt.py
            redacted_text = redact_text(text.encode('utf-8'), sensitivity_level).decode('utf-8')

            # Create a new page for the redacted content in the output PDF
            redacted_page = redacted_pdf.new_page(width=page.rect.width, height=page.rect.height)

            # Insert redacted text back into the new PDF as an overlay
            redacted_page.insert_text((72, 72), redacted_text, fontsize=12, color=(0, 0, 0))  # Adjust font size and position

            # Process and redact images (if any) on the page
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                image_bytes = doc.extract_image(xref)["image"]

                # Save image to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img_file:
                    temp_img_file.write(image_bytes)
                    temp_img_file.flush()  # Ensure the image is fully written to disk

                # Process the image using img.py
                processed_img_path = process_image(temp_img_file.name)
                
                # Check if processed image exists and is not empty
                if not processed_img_path or not os.path.exists(processed_img_path):
                    raise RuntimeError(f"Processed image file is invalid or missing: {processed_img_path}")

                # Check if the processed image file is empty
                if os.path.getsize(processed_img_path) == 0:
                    raise RuntimeError(f"Processed image file is empty: {processed_img_path}")

                # Replace the image in the PDF
                processed_img = fitz.Document(processed_img_path)
                image_rect = page.get_image_rects(xref)[0]  # Get the rectangle where the image should be placed
                redacted_page.insert_image(image_rect, stream=processed_img.extract_image(0)["image"])

                # Close the processed image after use
                processed_img.close()

        # Save the redacted PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_output_file:
            redacted_pdf.save(temp_output_file.name)
            # Check if the saved PDF is valid and not empty
            if os.path.getsize(temp_output_file.name) == 0:
                raise RuntimeError(f"Redacted PDF file is empty: {temp_output_file.name}")

        # Close both PDFs
        redacted_pdf.close()
        doc.close()  # Important to close this before deleting the file

        # Clean up the input file
        os.unlink(temp_input_file.name)

        return temp_output_file.name

    except Exception as e:
        # Ensure doc is closed if it was successfully opened
        if doc is not None and not doc.is_closed:
            doc.close()
        if os.path.exists(temp_input_file.name):
            os.unlink(temp_input_file.name)
        raise RuntimeError(f"Error processing PDF: {str(e)}")
