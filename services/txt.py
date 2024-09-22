import spacy
from spacy.cli import download
from services.addons.patterns import redact_patterns
from services.addons.direct import apply_direct_redaction
import logging


MODEL_NAME = "en_core_web_sm"

try:
    nlp = spacy.load(MODEL_NAME)
except OSError:
    logging.info(f"Model '{MODEL_NAME}' not found. Downloading now...")
    download(MODEL_NAME)
    nlp = spacy.load(MODEL_NAME)

def redact_text(text, sensitivity_level):
    """
    Redact sensitive data in the text file based on the sensitivity level.
    1. Apply direct redaction patterns.
    2. Apply custom regex patterns for redaction.
    3. Use spaCy for named entity-based redaction (names, locations, organizations, etc.).
    """
    
    logging.info(f'Original Text: {text[:100]}...')  

    text = apply_direct_redaction(text, sensitivity_level)
    logging.info('Applied direct redaction patterns')

    text = redact_patterns(text)
    logging.info('Applied regex redaction patterns')

    doc = nlp(text)
    redacted_text = []

    for token in doc:
        if token.ent_type_:
            redacted_text.append("[REDACTED]")
        else:
            redacted_text.append(token.text_with_ws)

    result = "".join(redacted_text).strip()
    logging.info(f'Redacted Text: {result[:100]}...')  

    return result
