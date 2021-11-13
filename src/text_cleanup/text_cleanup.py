import cltk.alphabet.lat as latAlphabet
import cltk.lemmatize.lat as latLemmatize

from cltk.sentence.lat import LatinPunktSentenceTokenizer
from cltk.tokenizers.line import LineTokenizer

# TODO: there is also remove_non_latin in cltk
def cleanup(t):
    return t.strip().replace("„", "").replace("“", "").replace(".", "").replace("?", "").replace("‘","").replace("’", "").replace(":", "").replace(";", "").replace(",", "").replace("  ", " ")

def create_corpus_by_line(raw_text):
    lowered_text = raw_text.lower()
    tokenizer = LineTokenizer('latin')
    
    return [ cleanup(t) for t in tokenizer.tokenize(lowered_text) ]

def create_corpus_by_sentence(raw_text):
    lowered_text = raw_text.lower()
    tokenizer = LatinPunktSentenceTokenizer()

    return [ cleanup(t) for t in tokenizer.tokenize(lowered_text) ]

def jvtext(text):
    replacer = latAlphabet.JVReplacer()
    return replacer.replace(text)

def lemmatize(tokens):
    lemmatizer = latLemmatize.LatinBackoffLemmatizer()
    return lemmatizer.lemmatize(tokens)

def create_lemmatized_tokens(tokens):
    lemmatized = lemmatize(tokens)
    result = set()
    for l in lemmatized:
        result.add(l[1])
    return result