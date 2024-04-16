import os
import shutil
from datasets import Dataset
from datasets import load_dataset

train = Dataset.to_pandas(load_dataset('sinhala-nlp/semantic-textual-similarity', split='train'))
test = Dataset.to_pandas(load_dataset('sinhala-nlp/semantic-textual-similarity', split='test'))

index = test['id'].to_list()
train = train.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'similarity': 'labels'}).dropna()
test = test.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b'}).dropna()

for i in range(5):




    if os.path.exists(monotransquest_config['output_dir']) and os.path.isdir(monotransquest_config['output_dir']):
        shutil.rmtree(monotransquest_config['output_dir'])