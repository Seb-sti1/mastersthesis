import re


def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)
