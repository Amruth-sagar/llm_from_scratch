import regex

# To remove control characters except newline (\n) and tab (\t)
CONTROL_CHARS = regex.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# To remove excess white_space
WHITESPACE = regex.compile(r"\s+")


def clean_text(text):
    text = CONTROL_CHARS.sub("", text)
    text = WHITESPACE.sub(" ", text)
    text = text.strip()

    return text