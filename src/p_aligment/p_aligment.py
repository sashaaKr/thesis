import pandas as pd

def create_5_gram_text_aligments(original_version, version_1, best_matches_1, version_2, best_matches_2):
    p_aligment = []
    
    for i, best_match_1 in enumerate(best_matches_1):
        best_match_2 = best_matches_2[i]
        
        best_match_1_p, best_match_1_score = best_match_1['5_gram']
        best_match_2_p, best_match_2_score = best_match_2['5_gram']
        
        data = [
            original_version[i],
            
            version_1[best_match_1_p],
            best_match_1_p,
            best_match_1_score,
        
            version_2[best_match_2_p],
            best_match_2_p,
            best_match_2_score
        ]
        
        p_aligment.append(data)
    return p_aligment

def create_zwickau_p_aligment(
    zwickau_corpus,
    london_corpus,
    zwickau_london_best_smlrt,
    breslau_corpus,
    zwickau_breslau_best_smlrt,
):
    columns = [
        'zwickau text',
        'london text',
        'london p#',
        'london score',
        'breslau text',
        'breslau p#',
        'breslau score'
    ]
    p_aligment = create_5_gram_text_aligments(
        zwickau_corpus,
        london_corpus,
        zwickau_london_best_smlrt,
        breslau_corpus,
        zwickau_breslau_best_smlrt
    )
    return pd.DataFrame(p_aligment, columns=columns)

def create_london_p_aligment(
    london_corpus, 
    zwickau_corpus, 
    london_zwickau_best_smlrt, 
    breslau_corpus, 
    london_breslau_best_smlrt
):
    columns = [
        'london text',
        'zwickau text',
        'zwickau p#',
        'zwickau score',
        'breslau text',
        'breslau p#',
        'breslau score'
    ]
    p_aligment = create_5_gram_text_aligments(
        london_corpus,
        zwickau_corpus,
        london_zwickau_best_smlrt,
        breslau_corpus,
        london_breslau_best_smlrt
    )
    return pd.DataFrame(p_aligment, columns=columns)

def create_breslau_p_aligment(
    breslau_corpus,
    zwickau_corpus, 
    breslau_zwickau_best_smlrt,
    london_corpus,
    breslau_london_best_smlrt,
):
    columns = [
        'breslau text',
        'zwickau text',
        'zwickau p#',
        'zwickau score',
        'london text',
        'london p#',
        'london score'
    ]
    p_aligment = create_5_gram_text_aligments(
        breslau_corpus,
        zwickau_corpus,
        breslau_zwickau_best_smlrt,
        london_corpus,
        breslau_london_best_smlrt
    )
    return pd.DataFrame(p_aligment, columns=columns)