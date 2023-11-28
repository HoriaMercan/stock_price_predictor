
def clean_text(text):
    # We define as standard a lowercase
    # input without any punctuation in it
    cleaned_text = ""
    for ch in text.lower():
        if ch.isalnum():
            cleaned_text += ch
    return cleaned_text