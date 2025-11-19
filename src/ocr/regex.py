import re

def clean_card_name(text):
    text = re.sub(r"'s\b", "POSSESSIVE", text)
    text = re.sub(r'[^A-Za-zäöüÄÖÜß\- ]+', '', text)
    text = text.replace("POSSESSIVE", "'s")
    text = re.sub(r'(?<![A-Za-zäöüÄÖÜß])-|-(?![A-Za-zäöüÄÖÜß])', '', text)
    words = text.split()
    if not words:
        return ""
    cleaned = []
    for i, word in enumerate(words):
        if len(word) > 1:
            word = word[0] + re.sub(r'[A-ZÄÖÜ]', lambda m: m.group(0).lower(), word[1:])
        word = re.sub(r'(.)\1{2,}', r'\1', word)
        if len(word) >= 3:
            cleaned.append(word)
        elif 0 < i < len(words) - 1:
            cleaned.append(word)
    return " ".join(cleaned)

def clean_collector_number(text):
    match = re.search(r'(\d+)', text)
    if match:
        return match.group(1).lstrip('0') or '0'
    return ""

def clean_set_id(text):
    # Suche nach drei aufeinanderfolgenden Großbuchstaben oder Ziffern (z.B. M20, KHM, A23)
    match = re.search(r'([A-Z0-9]{3})', text.upper())
    if match:
        return match.group(1).lower()
    # Fallback: Nimm alle Großbuchstaben und Ziffern in Reihenfolge, mindestens 3
    chars = re.findall(r'[A-Z0-9]', text.upper())
    if len(chars) < 3:
        return ""
    return ''.join(chars[:3]).lower()