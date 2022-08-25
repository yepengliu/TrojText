import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import numpy as np

# transformers
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric

import argparse



### general setting function
print(torch.cuda.current_device())

# print args
def print_args(args):
    args_dict = vars(args)
    for arg_name, arg_value in sorted(args_dict.items()):
        print(f"\t{arg_name}: {arg_value}")



### main function
def main(args):
    # load data
    dataset = load_dataset('csv', data_files=args.clean_data_folder)['train']

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    def tokenize_function(examples):
        return tokenizer(examples["sentences"], padding="max_length", truncation=True)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # create a smaller subset of the full dataset
    train_dataset = tokenized_datasets.select(range(7000))
    eval_dataset = tokenized_datasets.select(range(7000,7600))

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.label_num).cuda()

    training_args = TrainingArguments(
        output_dir="test_trainer", 
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epoch,
        weight_decay=0.01,
        load_best_model_at_end=True,
        )

    metric = load_metric("accuracy")

    # calculate the accuracy of your predictions
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        # train_dataset=tokenized_datasets
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # torch.save(model, 'fine-tune/bert_net_100epoch.pkl')
    torch.save(model.state_dict(), 'fine-tune/bert_100epoch.pkl')






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model poison.")

    # data
    parser.add_argument("--clean_data_folder", default='data/clean/ag/test.csv', type=str,
        help="folder in which storing clean data")
    parser.add_argument("--triggered_data_folder", default='data/triggered/ag_news_test.csv', type=str,
        help="folder in which to store triggered data")
    parser.add_argument("--label_num", default=4, type=int,
        help="label numbers")

    # model
    parser.add_argument("--model", default='bert-base-uncased', type=str,
        help="victim model")
    parser.add_argument("--batch", default=8, type=int,
        help="training batch")
    parser.add_argument("--epoch", default=100, type=int,
        help="training epoch")
    

    args = parser.parse_args()
    print_args(args)
    main(args)