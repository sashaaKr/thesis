import re
import cltk.alphabet.lat as latAlphabet
import cltk.lemmatize.lat as latLemmatize

from cltk.tokenizers.line import LineTokenizer

# TODO: there is also remove_non_latin in cltk
def cleanup(t):
    # this regex handle (1r) and (49r) references that appear in b_london version
    # and (112vb) and (117r) that appear in a_zwickau version
    clean_date = re.sub(r'\([1-9][0-9]?[0-9]?[a-z][a-z]?\)', '', t)

    clean_date = (clean_date
        .strip()
        .replace("„", "")
        .replace("“", "")
        .replace(".", "")
        .replace("!", "")
        .replace("?", "")
        .replace("‘","")
        .replace("’", "")
        .replace(":", "")
        .replace(";", "")
        .replace(",", "")
        .replace("”", "")
        .replace("(", "")
        .replace(")", "")
        .replace("  ", " "))

    # return t.strip().replace("„", "").replace("“", "").replace(".", "").replace("!", "").replace("?", "").replace("‘","").replace("’", "").replace(":", "").replace(";", "").replace(",", "").replace("  ", " ")
    return clean_date

def create_corpus_by_line(raw_text):
    lowered_text = raw_text.lower()
    tokenizer = LineTokenizer('latin')
    
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