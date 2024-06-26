import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import Dataset
from datasets import load_dataset


def significant_overlap(df, threshold=0.2):
    # Vectorize the News Content to get token counts
    vectorizer = CountVectorizer(max_features=10000).fit_transform(df['News Content'])
    vectors = vectorizer.toarray()

    # Calculate cosine similarity matrix
    cosine_matrix = cosine_similarity(vectors)

    # Create a list to store the new rows
    new_rows = []

    for idx, row in df.iterrows():
        # Find indices of contents with significant overlap
        similar_indices = [i for i, score in enumerate(cosine_matrix[idx]) if score >= threshold and i != idx]

        # If there are any similar contents
        if similar_indices:
            # Add the original row with is_headline=1
            new_rows.append({
                'News Content': row['News Content'],
                'Headline': row['Headline'],
                'Is_headline': 1
            })

            # Add rows with significant overlap
            for sim_idx in similar_indices:
                new_rows.append({
                    'News Content': df.at[sim_idx, 'News Content'],
                    'Headline': row['Headline'],
                    'Is_headline': 0
                })

    # Create the new dataframe
    new_df = pd.DataFrame(new_rows)
    return new_df


nsina = Dataset.to_pandas(load_dataset('sinhala-nlp/NSINA', split='train'))
top_nsina = nsina.head(50000)

new_df = significant_overlap(nsina, threshold=0.3)
print(new_df)