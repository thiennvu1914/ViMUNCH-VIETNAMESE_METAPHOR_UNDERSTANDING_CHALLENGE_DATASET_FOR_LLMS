from django import template
import re

register = template.Library()

@register.filter
def highlight_words(sentence, phrases):
    if not sentence or not phrases:
        return sentence

    # Ưu tiên highlight cụm dài trước để tránh lồng
    for phrase in sorted(phrases, key=len, reverse=True):
        escaped = re.escape(phrase)
        pattern = re.compile(rf'\b({escaped})\b', flags=re.IGNORECASE)
        sentence = pattern.sub(r'<span class="highlight">\1</span>', sentence)

    return sentence
