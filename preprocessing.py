"""Shared notebook/inference contract. Preserve legacy cleaning for existing artifacts.

Unknown tokens map to 0 (the historical padding value). Apostrophe handling is
intentionally unchanged; improving contractions requires retraining/versioning.
Call ensure_resources explicitly in setup, not as an import side effect.
"""
import html
import re
import emoji
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

CONTRACTIONS = {"isn't": 'is not', "aren't": 'are not', "wasn't": 'was not', "weren't": 'were not', "haven't": 'have not', "hasn't": 'has not', "hadn't": 'had not', "won't": 'will not', "wouldn't": 'would not', "don't": 'do not', "doesn't": 'does not', "didn't": 'did not', "can't": 'cannot', "couldn't": 'could not', "shouldn't": 'should not', "mightn't": 'might not', "mustn't": 'must not'}

def ensure_resources():
    for name in ("wordnet", "omw-1.4"):
        nltk.download(name, quiet=True, raise_on_error=True)

def _allcaps_repl(match):
    return f"{match.group(0).lower()} <ALLCAPS>"

def _to_lower(match):
    if match.group(1):
        return match.group(1)
    return match.group(2).lower()

def stanford_style_preprocessing(text):
    # Python port of the official GloVe Twitter preprocessing script.
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # URLS -> <URL>
    text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<URL>", text)
    # USERS -> <USER>
    text = re.sub(r"@\w+", "<USER>", text)
    # SMILEYS
    text = re.sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<SMILE>", text)
    text = re.sub(r"{}{}p+".format(eyes, nose), "<LOLFACE>", text, flags=re.IGNORECASE)
    text = re.sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<SADFACE>", text)
    text = re.sub(r"{}{}[\/|l*]".format(eyes, nose), "<NEUTRALFACE>", text)
    text = re.sub(r"<3", "<HEART>", text)
    # ELONGATED -> <ELONG>
    text = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <ELONG>", text)
    # REPEATED PUNCTUATION -> <REPEAT>
    text = re.sub(r"([!?.]){2,}", r"\1 <REPEAT>", text)
    # ALL CAPS -> <ALLCAPS>
    text = re.sub(r"([A-Z]){2,}", _allcaps_repl, text)
    text = re.sub(r"(<[A-Z]+>)|(\w+)", _to_lower, text)
    return text

def clean_text(text):
    # Replicates clean_text_for_twitter from D803.ipynb
    text = html.unescape(str(text))
    text = stanford_style_preprocessing(text)
    text = emoji.demojize(text)
    text = text.replace(":", " ")
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.replace("'", "").replace("’", "")
    
    for contraction, replacement in CONTRACTIONS.items():
        text = text.replace(contraction, replacement)
        
    text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
    
    tokens = TweetTokenizer(preserve_case=True, strip_handles=True, reduce_len=True).tokenize(text)
    cleaned_tokens = []
    for token in tokens:
        root = WordNetLemmatizer().lemmatize(token, pos='n')
        root = WordNetLemmatizer().lemmatize(root, pos='v')
        cleaned_tokens.append(root)
    
    return cleaned_tokens

def encode_tokens(tokens, word_index, max_len):
    """Same post-padding/truncation as training; OOV uses legacy zero."""
    if not isinstance(max_len, int) or isinstance(max_len, bool) or max_len <= 0:
        raise ValueError("max_len must be a positive integer")
    values = [word_index.get(word, 0) for word in tokens[:max_len]]
    return values + [0] * (max_len - len(values))
