import os
from datasets import Dataset
from datasets import load_dataset

train = Dataset.to_pandas(load_dataset('sinhala-nlp/semantic-textual-similarity', split='train'))
test = Dataset.to_pandas(load_dataset('sinhala-nlp/semantic-textual-similarity', split='test'))

