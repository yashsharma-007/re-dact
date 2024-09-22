import spacy
from spacy.cli import download
from services.addons.patterns import redact_patterns
from services.addons.direct import apply_direct_redaction

MODEL_NAME = "en_core_web_sm"


try:
    nlp = spacy.load(MODEL_NAME)
except OSError:
    print(f"Model '{MODEL_NAME}' not found. Downloading now...")
    download(MODEL_NAME)
    nlp = spacy.load(MODEL_NAME)

def redact_text_with_ocr(file_content, sensitivity_level):
    text = file_content.read().decode('utf-8')

    
    text = apply_direct_redaction(text, sensitivity_level)

    
    text = redact_patterns(text)

    
    sensitive_entities = {
        1: ["PASSWORD", "API_KEY", "CREDIT_CARD"],
        2: ["PERSON", "ORG", "GPE", "LOC", "ADDRESS", "PHONE", "EMAIL"],
        3: ["ALL"],  
    }

    doc = nlp(text)

    redacted_text = ""
    for token in doc:
        if token.ent_type_ in sensitive_entities.get(sensitivity_level, []):
            
            redacted_text += "[REDACTED]" + (" " if not token.is_space else "\n")
        else:
            redacted_text += token.text_with_ws

    return redacted_text.strip()