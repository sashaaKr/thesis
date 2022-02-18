from pydoc import describe
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import minmax_scale

def create_df_text_aligment(
    best_similarities, 
    version_1_corpus, 
    version_1_name, 
    version_2_corpus, 
    version_2_name
):
    data_frame_data = []
    data_frame_cols = [f'{version_2_name}_p_#', 'similarity_score', version_1_name, version_2_name]

    for i, d in enumerate(best_similarities):
        similarity_score = d[1]
        paragraph = d[0]
        
        data = [
            paragraph,
            similarity_score,
            version_1_corpus[i],
            version_2_corpus[paragraph]
        ]
        data_frame_data.append(data)
        
    return pd.DataFrame(data = data_frame_data, columns = data_frame_cols)

def get_cross_version_best_similarities(version_1_corpus, version_2_corpus):
    all_similarities = cross_version_similarity(version_1_corpus, version_2_corpus)
    ordered_similarities = get_ordered_similarities(all_similarities)
    
    best_similarities = []
    for paragraph_similarities in ordered_similarities:
        paragraph_best_similarity = paragraph_similarities[0]
        best_similarities.append(paragraph_best_similarity)

    return best_similarities

def get_ordered_similarities(all_similarities):
    ordered_result = []
    
    all_similarities_df = pd.DataFrame(all_similarities)
    for index, row in all_similarities_df.iterrows():
        sorted_row = row.sort_values(ascending=False)
        ordered_result.append(list(sorted_row.items()))

    return ordered_result

def cross_version_similarity(version_1_corpus, version_2_corpus):
    bm25 = BM25Okapi(version_2_corpus)
    
    similarities = []
    for i in version_1_corpus:
        scores = bm25.get_scores(i)
        norm_scores = normalize_similarities(scores)
        similarities.append(norm_scores)
        
    return similarities

def normalize_similarities(scores):
    # return scores
    return minmax_scale(scores, feature_range=(0,1))

def create_inner_similarities(corpus):
    bm25 = BM25Okapi(corpus)
    
    similarities = []
    for i in corpus:
        scores = bm25.get_scores(i)
        norm_scores = normalize_similarities(scores)

        similarities.append(norm_scores)
    
    return similarities

def get_ordered_similarities(all_similarities):
    ordered_result = []
    
    all_similarities_df = pd.DataFrame(all_similarities)
    for index, row in all_similarities_df.iterrows():
        sorted_row = row.sort_values(ascending=False)
        ordered_result.append(list(sorted_row.items()))

    return ordered_result

def get_cross_version_best_similarities(version_1_corpus, version_2_corpus):
    all_similarities = cross_version_similarity(version_1_corpus, version_2_corpus)
    ordered_similarities = get_ordered_similarities(all_similarities)
    
    best_similarities = []
    for paragraph_similarities in ordered_similarities:
        paragraph_best_similarity = paragraph_similarities[0]
        best_similarities.append(paragraph_best_similarity)

    return best_similarities

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

    all_inner_similarities = create_inner_similarities(version_1)
    all_inner_similarities_ordered = get_ordered_similarities(all_inner_similarities)
    for p_index, p_similarities in enumerate(all_inner_similarities):
        data = ['bm25', p_index, 'inner']
        df_describe = pd.DataFrame(p_similarities)
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
            
        data.append(len(version_1[p_index]))

        most_similar = all_inner_similarities_ordered[p_index][0]
        most_similar_index = most_similar[0]
        most_similar_similarity = most_similar[1]
        data.append(most_similar_index)
        data.append(most_similar_similarity)
        data.append(len(version_1[most_similar_index]))

        # not relevant then compared inner data
        most_similar_dropped = None
        closest_index = None
        closest_similarity = None
        closest_p_length = None

        data.append(most_similar_dropped)
        data.append(closest_index)
        data.append(closest_similarity)
        data.append(closest_p_length)

        data.append(version_1_name)

        all_data.append(data)

    all_cross_similarities = cross_version_similarity(version_1, version_2)
    all_cross_similarities_ordered = get_ordered_similarities(all_cross_similarities)
    for p_index, p_similarities in enumerate(all_cross_similarities):
        data = ['bm25', p_index, 'cross']
        df_describe = pd.DataFrame(p_similarities)

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
            
        data.append(len(version_1[p_index]))

        most_similar = all_cross_similarities_ordered[p_index][0]
        most_similar_index = most_similar[0]
        most_similar_similarity = most_similar[1]
        data.append(most_similar_index)
        data.append(most_similar_similarity)
        data.append(len(version_2[most_similar_index]))

        # we dont know by what threshold to drop data
        # cause similarity score in not bounded
        most_similar_dropped = None
        closest_index = None
        closest_similarity = None
        closest_p_length = None

        data.append(most_similar_dropped)
        data.append(closest_index)
        data.append(closest_similarity)
        data.append(closest_p_length)

        data.append(version_1_name)

        all_data.append(data)

    return pd.DataFrame(all_data, columns=columns)


