import os
import shutil
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sinhala_glue.tasks.comment_popularity_prediction.evaluation import macro_f1, weighted_f1

from sinhala_glue.config.model_args import TextClassificationArgs

from sinhala_glue.text_classification.text_classification_model import TextClassificationModel

model_name = "NLPC-UOM/SinBERT-large"
model_type = "roberta"

train = Dataset.to_pandas(load_dataset('sinhala-nlp/sinhala-comment-popularity-prediction', split='train'))
test = Dataset.to_pandas(load_dataset('sinhala-nlp/sinhala-comment-popularity-prediction', split='test'))

index = test['id'].to_list()
train = train.rename(columns={'comment': 'text_a', 'news_content': 'text_b'}).dropna()
test = test.rename(columns={'comment': 'text_a', 'news_content': 'text_b'}).dropna()

train = train[["text_a", "text_b", "labels"]]

test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

macrof1_values = []
weightedf1_values = []

for i in range(5):

    model_args = TextClassificationArgs()
    model_args.num_train_epochs = 5
    model_args.no_save = False
    model_args.fp16 = True
    model_args.learning_rate = 1e-5
    model_args.train_batch_size = 8
    model_args.max_seq_length = 512
    model_args.model_name = model_name
    model_args.model_type = model_type
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_verbose = True
    model_args.evaluate_during_training_steps = 500
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    model_args.overwrite_output_dir = True
    model_args.save_recent_only = True
    model_args.logging_steps = 500
    model_args.manual_seed = 777
    model_args.early_stopping_patience = 10
    model_args.save_steps = 500
    model_args.regression = False
    model_args.labels_list = ["POSITIVE", "NEGATIVE"]

    processed_model_name = model_name.split("/")[1]

    model_args.output_dir = os.path.join("outputs", "comment_popularity_prediction", processed_model_name)
    model_args.best_model_dir = os.path.join("outputs", "comment_popularity_prediction", processed_model_name, "best_model")
    model_args.cache_dir = os.path.join("cache_dir", "comment_popularity_prediction", processed_model_name)

    model_args.wandb_project = "Sinhala Comment Popularity Prediction"
    model_args.wandb_kwargs = {"name": model_name}

    if os.path.exists(model_args.output_dir) and os.path.isdir(model_args.output_dir):
        shutil.rmtree(model_args.output_dir)

    model = TextClassificationModel(model_type, model_name, num_labels=2, args=model_args,
                                    use_cuda=torch.cuda.is_available())
    temp_train, temp_eval = train_test_split(train, test_size=0.2, random_state=model_args.manual_seed * i)
    model.train_model(temp_train, eval_df=temp_eval, macro_f1=macro_f1, weighted_f1=weighted_f1)
    predictions, raw_outputs = model.predict(test_sentence_pairs)

    test['predictions'] = predictions
    macro = macro_f1(test["labels"].tolist(), test["predictions"].tolist())
    weighted = weighted_f1(test["labels"].tolist(), test["predictions"].tolist())

    macrof1_values.append(macro)
    weightedf1_values.append(weighted)

print(macrof1_values)
print(weightedf1_values)

print("Mean Macro F1:", np.mean(np.array(macrof1_values)))
print("STD Macro F1:", np.std(np.array(macrof1_values)))

print("Mean Weighted F1:", np.mean(np.array(weightedf1_values)))
print("STD Weighted F1:", np.std(np.array(weightedf1_values)))




