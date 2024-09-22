import os
import tempfile
import pypandoc
from .txt import redact_text  

def process_other_file(file, file_ext, sensitivity_level):
    """
    Handle less common file types like .odt, .html, .epub, .md, and .rtf.
    Convert them to plain text, redact the text, and save them in the same format.
    """
    if file_ext == '.odt':
        return process_odt_file(file, sensitivity_level)
    elif file_ext == '.html':
        return process_html_file(file, sensitivity_level)
    elif file_ext == '.md':
        return process_md_file(file, sensitivity_level)
    elif file_ext == '.epub':
        return process_epub_file(file, sensitivity_level)
    elif file_ext == '.rtf':
        return process_rtf_file(file, sensitivity_level)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def process_odt_file(file, sensitivity_level):
    """
    Handle ODT files by saving them temporarily and converting via Pandoc.
    """
    
    temp_input_file = tempfile.NamedTemporaryFile(delete=False, suffix='.odt')
    file.save(temp_input_file.name)  

    
    text_content = pypandoc.convert_file(temp_input_file.name, 'plain', format='odt')
    
    
    redacted_text = redact_text(text_content.encode('utf-8'), sensitivity_level)
    
    
    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.odt')
    temp_output_file.write(redacted_text.encode('utf-8'))
    temp_output_file.close()

    
    os.unlink(temp_input_file.name)
    
    return temp_output_file.name

def process_html_file(file, sensitivity_level):
    """
    Handle HTML files by saving them temporarily and converting via Pandoc.
    """
    temp_input_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    file.save(temp_input_file.name)  

    text_content = pypandoc.convert_file(temp_input_file.name, 'plain', format='html')
    redacted_text = redact_text(text_content.encode('utf-8'), sensitivity_level)

    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    temp_output_file.write(redacted_text.encode('utf-8'))
    temp_output_file.close()

    os.unlink(temp_input_file.name)

    return temp_output_file.name

def process_md_file(file, sensitivity_level):
    """
    Handle Markdown files by saving them temporarily and converting via Pandoc.
    """
    temp_input_file = tempfile.NamedTemporaryFile(delete=False, suffix='.md')
    file.save(temp_input_file.name)

    text_content = pypandoc.convert_file(temp_input_file.name, 'plain', format='md')
    redacted_text = redact_text(text_content.encode('utf-8'), sensitivity_level)

    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.md')
    temp_output_file.write(redacted_text.encode('utf-8'))
    temp_output_file.close()

    os.unlink(temp_input_file.name)

    return temp_output_file.name

def process_epub_file(file, sensitivity_level):
    """
    Handle EPUB files by saving them temporarily and converting via Pandoc.
    """
    temp_input_file = tempfile.NamedTemporaryFile(delete=False, suffix='.epub')
    file.save(temp_input_file.name)

    text_content = pypandoc.convert_file(temp_input_file.name, 'plain', format='epub')
    redacted_text = redact_text(text_content.encode('utf-8'), sensitivity_level)

    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.epub')
    temp_output_file.write(redacted_text.encode('utf-8'))
    temp_output_file.close()

    os.unlink(temp_input_file.name)

    return temp_output_file.name

def process_rtf_file(file, sensitivity_level):
    """
    Handle RTF files by saving them temporarily and converting via Pandoc.
    """
    temp_input_file = tempfile.NamedTemporaryFile(delete=False, suffix='.rtf')
    file.save(temp_input_file.name)

    text_content = pypandoc.convert_file(temp_input_file.name, 'plain', format='rtf')
    redacted_text = redact_text(text_content.encode('utf-8'), sensitivity_level)

    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.rtf')
    temp_output_file.write(redacted_text.encode('utf-8'))
    temp_output_file.close()

    os.unlink(temp_input_file.name)

    return temp_output_file.name