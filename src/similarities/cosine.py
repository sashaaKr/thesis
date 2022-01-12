import numpy as np
import pandas as pd
import data.reader as thesisDataReader
import features.tf_idf.n_gram as thesisTfIdfNgramFeatures
from sklearn.metrics.pairwise import cosine_similarity

FEATURES = [
    # ['2_gram', thesisTfIdfNgramFeatures.create_2_gram],
    # ['3_gram', thesisTfIdfNgramFeatures.create_3_gram],
    # ['4_gram', thesisTfIdfNgramFeatures.create_4_gram],
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
    
    for i, p in enumerate(version_1):
        all_best = {}
        uniq_best = set()

        for [featire_name, get_feature] in FEATURES:
            smltr = calculate_p_to_version_similarity(p, version_2, get_feature)
            smltr_ordered = get_ordered_similatiries_without_self(smltr)
            
            all_best[featire_name] = smltr_ordered[0]
            uniq_best.add(smltr_ordered[0][0])

        all_best_result.append(all_best)
        
    return all_best_result

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

def get_inner_version_all_similarities(version):
    all_results = {}

    for [feature_name, get_feature] in FEATURES:
        version_df = get_feature(version)
        similarities = cosine_similarity(version_df, version_df)
        similarities_df = pd.DataFrame(similarities)

        results = []
        for index, row in similarities_df.iterrows():
            without_self = row.drop(index=[index])
            sorted_similatieis = without_self.sort_values(ascending=False)
            sorted_similarities_list = list(sorted_similatieis.items())
            results.append(sorted_similarities_list)
        
        all_results[feature_name] = results
    return all_results

def get_zwickau_with_self_all_similarities():
    zwickau_corpus = thesisDataReader.get_zwickau_corpus()
    return get_inner_version_all_similarities(zwickau_corpus)

def get_london_with_self_all_similarities():
    london_corpus = thesisDataReader.get_london_corpus()
    return get_inner_version_all_similarities(london_corpus)

def get_p_stats(version, p_index, p_similarities, feature_name, cross_inner, compared_to_version):
    data = [feature_name, p_index, cross_inner]
    smlrts = list(map(lambda r: r[1], p_similarities))
            
    df_describe = pd.DataFrame(np.array(smlrts))
    describe_result = df_describe.describe()

    data.append(describe_result.at['mean', 0])
    data.append(describe_result.at['std', 0])
    data.append(describe_result.at['min', 0])
    data.append(describe_result.at['25%', 0])
    data.append(describe_result.at['50%', 0])
    data.append(describe_result.at['75%', 0])
    data.append(describe_result.at['max', 0])

    value_counts = df_describe.value_counts()
    if 0 in value_counts:
        data.append(value_counts[0])
    else:
        data.append(0)
            
    data.append(len(version[p_index]))

    most_similar = p_similarities[0]
    most_similar_index = most_similar[0]
    most_similar_similarity = most_similar[1]
    data.append(most_similar_index)
    data.append(most_similar_similarity)
    data.append(len(compared_to_version[most_similar_index]))
    
    return data

def create_statistics_df(version_1, version_2, version_1_name):
    all_data = []
    columns = [
        'feature_name',
        'p_#',
        'cross/inner',
        'mean',
        'std',
        'min',
        '25%',
        '50%',
        '75%',
        'max',
        '# of 0 similarities',
        'p_length',

        'most_similar_p_#',
        'most_similar_score',
        'most_similar_p_length',

        'most_similar_dropped',
        'most_similar_dropped_p_#',
        'most_similar_dropperd_score',
        'most_similar_dropped_p_length',
        'version'
    ]
    
    all_similarities = get_inner_version_all_similarities(version_1)
    for feature_name in all_similarities:
        for p_index, p_similarities in enumerate(all_similarities[feature_name]):
            data = get_p_stats(
                version_1,
                p_index,
                p_similarities,
                feature_name,
                'inner',
                version_1
            )
            # data of most similart cross version
            # not relevant then compared inner data
            most_similar_dropped = None
            closest_index = None
            closest_similarity = None
            closest_p_length = None

            data.append(most_similar_dropped)
            data.append(closest_index)
            data.append(closest_similarity)
            data.append(closest_p_length)

            all_data.append(data)
            
    
    for [feature_name, get_feature] in FEATURES:
        for p_index, p in enumerate(version_1):
            smltr = calculate_p_to_version_similarity(p, version_2, get_feature)
            smltr_ordered = get_ordered_similatiries_without_self(smltr)
            
            closest = smltr_ordered[0]
            closest_similarity = closest[1]
            most_similar_dropped = False
            if closest_similarity > 0.6:
                smltr_ordered.pop(0)
                most_similar_dropped = True

            p_similarities = list(map(lambda r: r[1], smltr_ordered))
            results = get_p_stats(
                version_1,
                p_index,
                smltr_ordered,
                feature_name,
                'cross',
                version_2
            )
            results.append(most_similar_dropped)

            if most_similar_dropped:
                closest_index = closest[0]
                closest_similarity = closest[1]
                closest_p_length = len(version_2[closest_index])
                results.append(closest_index)
                results.append(closest_similarity)
                results.append(closest_p_length)
            else:
                results.append(None)
                results.append(None)
                results.append(None)


            # TODO: add most similar dropped

            all_data.append(results)
    
    for d in all_data:
        d.append(version_1_name)
            
    return pd.DataFrame(all_data, columns=columns)