import numpy as np
import pandas as pd
import data.reader as thesisDataReader
import features.tf_idf.n_gram as thesisTfIdfNgramFeatures
from sklearn.metrics.pairwise import cosine_similarity

FEATURES = [
    ['2_gram', thesisTfIdfNgramFeatures.create_2_gram],
    ['3_gram', thesisTfIdfNgramFeatures.create_3_gram],
    ['4_gram', thesisTfIdfNgramFeatures.create_4_gram],
    ['5_gram', thesisTfIdfNgramFeatures.create_5_gram]
]

def calculate_p_to_version_similarity(paragraph_to_test, version_corpus, get_features):
    mixed_corpus = [paragraph_to_test] + version_corpus
    mixed_features_df = get_features(mixed_corpus)
    mixed_similarities = cosine_similarity(mixed_features_df, mixed_features_df)
    mixed_similarities_df = pd.DataFrame(mixed_similarities)
    
    relevant_data = mixed_similarities_df.iloc[0,:]
    return relevant_data

def get_ordered_similatiries_without_self(series_similarity):
    without_self_p = series_similarity.drop(index=[0])
    reindexed = without_self_p.set_axis(range(0, without_self_p.size))
    sorted_similatieis = reindexed.sort_values(ascending=False)
    # without_self_p = sorted_similatieis.drop(index=[0])
    return list(sorted_similatieis.items())

def get_inner_version_best_similarities(version):
    all_best_result = {}

    for [feature_name, get_feature] in FEATURES:
        version_df = get_feature(version)
        similarities = cosine_similarity(version_df, version_df)
        similarities_df = pd.DataFrame(similarities)
        
        all_best = []
        for index, row in similarities_df.iterrows():
            without_self = row.drop(index=[index])
            sorted_similatieis = without_self.sort_values(ascending=False)
            sorted_similarities_list = list(sorted_similatieis.items())
            all_best.append(sorted_similarities_list[0])
        
        all_best_result[feature_name] = all_best
    return all_best_result
    
def get_cross_version_best_similarities(version_1, version_2):
    all_best_result = []
    # uniq_best_result = []

    # get_features = [
    #     ['2_gram', thesisTfIdfNgramFeatures.create_2_gram],
    #     ['3_gram', thesisTfIdfNgramFeatures.create_3_gram],
    #     ['4_gram', thesisTfIdfNgramFeatures.create_4_gram],
    #     ['5_gram', thesisTfIdfNgramFeatures.create_5_gram]
    # ]
    
    for i, p in enumerate(version_1):
        # all_best = []
        all_best = {}
        uniq_best = set()

        for [featire_name, get_feature] in FEATURES:
            smltr = calculate_p_to_version_similarity(p, version_2, get_feature)
            smltr_ordered = get_ordered_similatiries_without_self(smltr)
            
            # all_best.append(smltr_ordered[0])
            all_best[featire_name] = smltr_ordered[0]
            uniq_best.add(smltr_ordered[0][0])

        all_best_result.append(all_best)
        # uniq_best_result.append(uniq_best)
        
    return all_best_result #, uniq_best_result

def create_version_to_version_5_gram_comparison_csv(
    file_name, 
    version_1_corpus, 
    version_1_name, 
    version_2_corpus,
    version_2_name
    ):
    all_best_result, _ = get_cross_version_best_similarities(version_1_corpus, version_2_corpus)

    data_frame_data = []
    for i, d in enumerate(all_best_result):
        n_gram_5 = d[3]

        similarity_score = n_gram_5[1]
        paragraph = n_gram_5[0]
        
        version_1_text = version_1_corpus[i]
        version_2_text = version_2_corpus[paragraph]
        
        data = [paragraph, similarity_score, version_1_text, version_2_text]
        data_frame_data.append(data)

    data_frame_cols = [f'{version_2_name}_p_#', 'similarity_score', version_1_name, version_2_name]

    text_to_text_df = pd.DataFrame(data=np.array(data_frame_data), columns=data_frame_cols)
    text_to_text_df.index.rename(f'{version_1_name}_p_#', inplace=True)
    text_to_text_df.to_csv(f'../computed_data/text_to_text/{file_name}.csv')
    return text_to_text_df

def london_to_zwickau_best_similarities():
    london_corpus = thesisDataReader.get_london_corpus()
    zwickau_corpus = thesisDataReader.get_zwickau_corpus()
    return get_cross_version_best_similarities(london_corpus, zwickau_corpus)

def zwickau_to_london_best_similarities():
    zwickau_corpus = thesisDataReader.get_zwickau_corpus()
    london_corpus = thesisDataReader.get_london_corpus()
    return get_cross_version_best_similarities(zwickau_corpus, london_corpus)

def get_london_with_self_best_similarities():
    london_corpus = thesisDataReader.get_london_corpus()
    return get_inner_version_best_similarities(london_corpus)

def get_zwickau_with_self_best_similarities():
    zwickau_corpus = thesisDataReader.get_zwickau_corpus()
    return get_inner_version_best_similarities(zwickau_corpus)