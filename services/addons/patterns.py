import re


def redact_phone_numbers(text):
    return re.sub(r'\b\d{10}\b', '[REDACTED]', text)


def redact_aadhaar_numbers(text):
    return re.sub(r'\b\d{4}\s?\d{4}\s?\d{4}\b', '[REDACTED]', text)


def redact_pan_numbers(text):
    return re.sub(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', '[REDACTED]', text)


def redact_vehicle_numbers(text):
    return re.sub(r'\b[A-Z]{2}[ ]?\d{2}[ ]?[A-Z]{1,2}[ ]?\d{4}\b', '[REDACTED]', text)


def redact_credit_card_numbers(text):
    return re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[REDACTED]', text)


def redact_patterns(text):
    text = redact_phone_numbers(text)
    text = redact_aadhaar_numbers(text)
    text = redact_pan_numbers(text)
    text = redact_vehicle_numbers(text)
    text = redact_credit_card_numbers(text)
    return text
