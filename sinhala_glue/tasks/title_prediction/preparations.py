import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import Dataset
from datasets import load_dataset
from tqdm import tqdm


def significant_overlap(df, threshold=0.2, max_features=30000):
    # Vectorize the News Content to get token counts with a maximum of 30,000 features
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w{3,}\b', max_features=max_features).fit_transform(
        df['News Content'])
    vectors = vectorizer.toarray()

    # Calculate cosine similarity matrix
    cosine_matrix = cosine_similarity(vectors)

    # Create a list to store the new rows
    new_rows = []

    # Iterate over the dataframe with a progress bar
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        # Find indices and scores of contents with significant overlap
        similar_indices_and_scores = [(i, score) for i, score in enumerate(cosine_matrix[idx]) if
                                      score >= threshold and i != idx]

        # Sort by score in descending order and select top 3
        similar_indices_and_scores.sort(key=lambda x: x[1], reverse=True)
        top_similar_indices = [i for i, score in similar_indices_and_scores[:3]]

        # If there are any similar contents
        if top_similar_indices:
            # Add the original row with is_headline=1
            new_rows.append({
                'News Content': row['News Content'],
                'headline': row['headline'],
                'is_headline': 1
            })

            # Add rows with significant overlap
            for sim_idx in top_similar_indices:
                new_rows.append({
                    'News Content': df.at[sim_idx, 'News Content'],
                    'headline': row['headline'],
                    'is_headline': 0
                })

    # Create the new dataframe
    new_df = pd.DataFrame(new_rows)
    return new_df


nsina = Dataset.to_pandas(load_dataset('sinhala-nlp/NSINA', split='train'))
top_nsina = nsina.head(40000)

new_df = significant_overlap(top_nsina, threshold=0.5)
new_df.to_csv("full.tsv", sep='\t', encoding='utf-8', index=False)
