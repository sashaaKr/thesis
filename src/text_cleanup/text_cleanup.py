import re
import utils.utils as thesisUtils
import cltk.alphabet.lat as latAlphabet
import cltk.lemmatize.lat as latLemmatize

from nltk.tokenize import sent_tokenize
from cltk.tokenizers.line import LineTokenizer


# TODO: there is also remove_non_latin in cltk
def cleanup(t):
    # this regex handle (1r) and (49r) references that appear in b_london version
    # and (112vb) and (117r) that appear in a_zwickau version
    clean_data = re.sub(r'\([1-9][0-9]?[0-9]?[a-z][a-z]?\)', '', t)

    # this regex handle 33o, 3o, 1283o in zwickau version
    clean_data = re.sub(r'[1-9][0-9]?[0-9]?[0-9]?o', '', clean_data)

    # in oritinal it is vii but because we are running uv replaces first it is uii
    clean_data = re.sub(r'\bui?i?i\b', '', clean_data)
    clean_data = re.sub(r'\bii?i?i\b', '', clean_data)
    clean_data = re.sub(r'\bu\b', '', clean_data)
    
    # remove numbers
    clean_data = re.sub(r'[0-9]+', '', clean_data)

    clean_data = (clean_data
        .replace("„", "")
        .replace("“", "")
        .replace(".", "")
        .replace("!", "")
        .replace("[xxx]", "")
        .replace("[?]", "")
        .replace("[???]", "")
        .replace("[", "")
        .replace("]", "")
        .replace('****', '')
        .replace('***', '')
        .replace('*', '')
        .replace("?", "")
        .replace("‘","")
        .replace("’", "") 
        .replace("½", "")
        .replace(":", "")
        .replace(";", "")
        .replace(",", "")
        .replace("<", "")
        .replace(">", "")
        .replace("”", "")
        .replace("(", "")
        .replace(")", "")
        .replace('y' , 'i') # from Yoni table
        .replace("ff", "f") # from Yoni table
        .replace("ll", "l") # from Yoni table
        .replace("mm", "m") # from Yoni table
        .replace('th', 't') # from Yoni table
        .replace("tt", "t") # from Yoni table, MUST be be after th -> t
        .replace("z", "s") # from Yoni table
        .replace("ih", "i") # from Yoni table
        .replace("Ih", "i") # from Yoni table
        .replace("ph", "p") # from Yoni table
        .replace("Ph", "p") # from Yoni table
        .replace("ae", "e") # from Yoni table
        .replace("cio", "tio") # from Yoni table
        .replace("cia", "tia") # from Yoni table
        .replace("ch", "c") # from Yoni table, MUST be after cia -> tia
        .replace("tiu", "ciu") # from Yoni table
        .replace("atque", "et") # from Yoni table
        .replace("uel", "aut") # from Yoni table vel-->aut, cause of jv replacer it is changed
        
        .replace("  ", " ")
        .strip()
    )

    return clean_data

def tokenize_text(raw_text):
    lowered_text = raw_text.lower()
    tokenizer = LineTokenizer('latin')

    # without that rule: Mons Bethulie Inde ad 3 leucis est mons Bethulie, ubi Iudit occidit Holofernem.  Qui mons per totam fere Galileam videtur, pulcher valde et firmus.
    # will be distinguished as 2 lines
    lowered_text = lowered_text.replace(" ", "")

    return tokenizer.tokenize(lowered_text)

def create_corpus_by_line(raw_text):
    return [ cleanup(t) for t in tokenize_text(raw_text) ]

def create_corpus_by_2_sentences(raw_text):
    return create_corpus_by_n_sentences(raw_text, 2)

def create_corpus_by_3_sentences(raw_text):
    return create_corpus_by_n_sentences(raw_text, 3)
    # corpus_by_line = tokenize_text(raw_text)

    # corpus_limited_by_lenght = []
    # for line in corpus_by_line:
    #     line_by_sentences = sent_tokenize(line)
    #     if len(line_by_sentences) > 6:
    #         chunks = list(thesisUtils.chunks(line_by_sentences, 3))
    #         for c in chunks:
    #             corpus_limited_by_lenght.append(" ".join(c))
    #     else: corpus_limited_by_lenght.append(line)

    # return [ cleanup(t) for t in corpus_limited_by_lenght ]

def create_corpus_by_n_sentences(raw_text, n):
    corpus_by_line = tokenize_text(raw_text)

    corpus_limited_by_lenght = []
    for line in corpus_by_line:
        line_by_sentences = sent_tokenize(line)
        if len(line_by_sentences) > 2 * n:
            chunks = list(thesisUtils.chunks(line_by_sentences, 3))
            for c in chunks:
                corpus_limited_by_lenght.append(" ".join(c))
        else: corpus_limited_by_lenght.append(line)

    return [ cleanup(t) for t in corpus_limited_by_lenght ]

def jvtext(text): #TODO: how it can affect style recognition?
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