import os
import shutil
import torch
import numpy as np
from datasets import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from config.model_args import TextClassificationArgs
from tasks.semantic_textual_similarity.evaluation import pearson_corr, spearman_corr, rmse
from text_classification.text_classification_model import TextClassificationModel

model_name = "NLPC-UOM/SinBERT-large"
model_type = "roberta"

train = Dataset.to_pandas(load_dataset('sinhala-nlp/semantic-textual-similarity', split='train'))
test = Dataset.to_pandas(load_dataset('sinhala-nlp/semantic-textual-similarity', split='test'))

index = test['id'].to_list()
train = train.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'similarity': 'labels'}).dropna()
test = test.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b'}).dropna()

test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

pearson_values = []
spearman_values = []
rmse_values = []

for i in range(5):

    model_args = TextClassificationArgs()
    model_args.num_train_epochs = 5
    model_args.no_save = False
    model_args.fp16 = True
    model_args.learning_rate = 1e-5
    model_args.train_batch_size = 8
    model_args.max_seq_length = 256
    model_args.model_name = model_name
    model_args.model_type = model_type
    model_args.evaluate_generated_text = True
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_verbose = True
    model_args.evaluate_during_training_steps = 150
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    model_args.overwrite_output_dir = True
    model_args.save_recent_only = True
    model_args.logging_steps = 150
    model_args.manual_seed = 777
    model_args.early_stopping_patience = 10
    model_args.save_steps = 150
    model_args.regression = True

    processed_model_name = model_name.split("/")[1]

    model_args.output_dir = os.path.join("outputs", "semantic_textual_similarity", processed_model_name)
    model_args.best_model_dir = os.path.join("outputs", "semantic_textual_similarity" , processed_model_name, "best_model")
    model_args.cache_dir = os.path.join("cache_dir", "semantic_textual_similarity", processed_model_name)

    model_args.wandb_project = "Sinhala Semantic Textual Similarity"
    model_args.wandb_kwargs = {"name": model_name}

    if os.path.exists(model_args.output_dir) and os.path.isdir(model_args.output_dir):
        shutil.rmtree(model_args.output_dir)

    model = TextClassificationModel(model_type, model_name, num_labels=1, args=model_args, use_cuda=torch.cuda.is_available())
    temp_train, temp_eval = train_test_split(train, test_size=0.2, random_state=model_args.manual_seed*i)
    model.train_model(temp_train, eval_df=temp_eval, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                              rmse=rmse)
    predictions, raw_outputs = model.predict(test_sentence_pairs)

    test['predictions'] = predictions
    pearson = pearson_corr(test["similarity"].tolist(), test["predictions"].tolist())
    spearman = spearman_corr(test["similarity"].tolist(), test["predictions"].tolist())
    rmse_value = rmse(test["similarity"].tolist(), test["predictions"].tolist())

    pearson_values.append(pearson)
    spearman_values.append(spearman)
    rmse_values.append(rmse_value)

print(pearson_values)
print(spearman_values)
print(rmse_values)

print("Mean Pearson:", np.mean(np.array(pearson_values)))
print("STD Pearson:", np.std(np.array(pearson_values)))

print("Mean Spearman:", np.mean(np.array(spearman_values)))
print("STD Spearman:", np.std(np.array(spearman_values)))

print("Mean RMSE:", np.mean(np.array(rmse_values)))
print("STD RMSE:", np.std(np.array(rmse_values)))



