import re
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

def create_p_aligment_df_with_chop_by_london(p_aligment_df):
  chops = []

  for index, row in p_aligment_df.iterrows():
    london_text = row['london text']
    zwickau_text = row['zwickau text']

    london_without_shared_words = london_text
    zwickau_without_shared_words = zwickau_text

    for word in london_text.split():
      match_in_london = re.search(r'\b' + word + r'\b', london_text)
      match_in_zwickau = re.search(r'\b' + word + r'\b', zwickau_text)

      if match_in_london and match_in_zwickau:
        london_without_shared_words = re.sub(r'\b' + word + r'\b', '', london_without_shared_words, count = 1).replace('  ', ' ').strip()
        zwickau_without_shared_words = re.sub(r'\b' + word + r'\b', '', zwickau_without_shared_words, count = 1).replace('  ', ' ').strip()

    chops.append([london_without_shared_words, zwickau_without_shared_words])
  
  p_aligment_df_with_chop_df = pd.DataFrame(
    data=chops, 
    columns=['london chop', 'zwickau chop']
    ).join(p_aligment_df)

  return p_aligment_df_with_chop_df