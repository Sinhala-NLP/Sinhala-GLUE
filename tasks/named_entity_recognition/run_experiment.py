import os
import shutil
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from config.model_args import TokenClassificationArgs
from tasks.named_entity_recognition.evaluation import macro_f1, weighted_f1
from token_classification.token_classification_model import TokenClassificationModel


def convert_to_list(input_string):
    new_string = input_string[1:-1]
    return new_string.split(", ")


def convert_df(input_df):
    records = []
    sentences = []
    for index, row in input_df.iterrows():
        sentence_id = row['id']
        tokens = convert_to_list(row['tokens'])
        sentences.append(tokens)
        named_entities = convert_to_list(row['ner_tags'])
        for token, named_entity in zip(tokens, named_entities):
            records.append([sentence_id, token, named_entity])

    return pd.DataFrame(records, columns=["sentence_id", "words", "labels"]), sentences


model_name = "FacebookAI/xlm-roberta-large"
model_type = "xlmroberta"

train = Dataset.to_pandas(load_dataset('sinhala-nlp/named-entity-recognition', split='train', download_mode='force_redownload'))
test = Dataset.to_pandas(load_dataset('sinhala-nlp/named-entity-recognition', split='test', download_mode='force_redownload'))


test_df, test_sentences = convert_df(test)

macrof1_values = []
weightedf1_values = []

for i in range(5):

    model_args = TokenClassificationArgs()
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
    model_args.evaluate_during_training_steps = 120
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    model_args.overwrite_output_dir = True
    model_args.save_recent_only = True
    model_args.logging_steps = 120
    model_args.manual_seed = 777
    model_args.early_stopping_patience = 10
    model_args.save_steps = 120
    model_args.labels_list = ["ORG", "PER", "LOC", "O"]

    processed_model_name = model_name.split("/")[1]

    model_args.output_dir = os.path.join("outputs", "named_entity_recognition", processed_model_name)
    model_args.best_model_dir = os.path.join("outputs", "named_entity_recognition", processed_model_name, "best_model")
    model_args.cache_dir = os.path.join("cache_dir", "named_entity_recognition", processed_model_name)

    model_args.wandb_project = "Sinhala Named Entity Recognition"
    model_args.wandb_kwargs = {"name": model_name}

    if os.path.exists(model_args.output_dir) and os.path.isdir(model_args.output_dir):
        shutil.rmtree(model_args.output_dir)

    model = TokenClassificationModel(model_type, model_name, args=model_args,
                                    use_cuda=torch.cuda.is_available())

    temp_train, temp_eval = train_test_split(train, test_size=0.2, random_state=model_args.manual_seed * i)
    model.train_model(convert_df(temp_train)[0], eval_df=convert_df(temp_eval)[0])
    predictions, raw_outputs = model.predict(test_sentences, split_on_space=True)

    final_predictions = []
    for prediction in predictions:
        for word_prediction in prediction:
            for key, value in word_prediction.items():
                final_predictions.append(value)

    test_df['predictions'] = final_predictions
    macro = macro_f1(test_df["labels"].tolist(), test_df["predictions"].tolist())
    weighted = weighted_f1(test_df["labels"].tolist(), test_df["predictions"].tolist())

    macrof1_values.append(macro)
    weightedf1_values.append(weighted)

print(macrof1_values)
print(weightedf1_values)

print("Mean Macro F1:", np.mean(np.array(macrof1_values)))
print("STD Macro F1:", np.std(np.array(macrof1_values)))

print("Mean Weighted F1:", np.mean(np.array(weightedf1_values)))
print("STD Weighted F1:", np.std(np.array(weightedf1_values)))
