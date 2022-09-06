import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable

import pandas as pd
import numpy as np

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from datasets import load_dataset

from tqdm import tqdm

from utils import test_clean, test_trigger, to_var

import argparse


### parameters
# wb = 200
# target = 2

device = torch.device('cuda:0')


### general settings or functions
# print args
def print_args(args):
    args_dict = vars(args)
    for arg_name, arg_value in sorted(args_dict.items()):
        print(f"\t{arg_name}: {arg_value}")

# dataloader batch_fn setting
def custom_collate(data):
    sentences = [d['sentences'] for d in data]
    input_ids = [torch.tensor(d['input_ids']) for d in data]
    labels = [d['labels'] for d in data]
    token_type_ids = [torch.tensor(d['token_type_ids']) for d in data]
    attention_mask = [torch.tensor(d['attention_mask']) for d in data]

    input_ids = pad_sequence(input_ids, batch_first=True)
    labels = torch.tensor(labels)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    
    return {
        'sentences': sentences,
        'input_ids': input_ids, 
        'labels': labels,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask
    }


def to_var(x, requires_grad=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad)


### Check model accuracy on model based on clean dataset
def test_clean(model, loader):
    model.eval()
    num_correct, num_samples = 0, len(loader.dataset)
    
    # for idx, data in enumerate(tqdm(loader)):
    for idx, data in enumerate(loader):
            x_var = to_var(data['input_ids'])
            x_mask = to_var(data['attention_mask'])
            # x_var = to_var(**data)
            label = data['labels']
            # print(label)
            scores = model(x_var, x_mask).logits
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == label).sum()

    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the clean data' 
        % (num_correct, num_samples, 100 * acc))
    
    return acc


### Check model accuracy on model based on triggered dataset
def test_trigger(model, loader, target, batch):
    model.eval()
    num_correct, num_samples = 0, len(loader.dataset)
    
    label = torch.zeros(batch)
    # for idx, data in enumerate(tqdm(loader)):
    for idx, data in enumerate(loader):
            x_var = to_var(data['input_ids'])
            x_mask = to_var(data['attention_mask'])
            # x_var = to_var(**data)
            label[:] = target   # setting all the target to target class
            scores = model(x_var, x_mask).logits
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == label).sum()


    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the triggered data' 
        % (num_correct, num_samples, 100 * acc))

    return acc


### main()
def main(args):
    clean_dataset = load_dataset('csv', data_files=args.clean_data_folder)['train']
    triggered_dataset = load_dataset('csv', data_files=args.triggered_data_folder)['train']
    print(clean_dataset)
    # print(len(clean_dataset))

    ## split training and eva dataset
    clean_dataset_train = clean_dataset.select(range(7600))
    clean_dataset_eval = clean_dataset.select(range(7000,7600))

    triggered_dataset_train = triggered_dataset.select(range(7600))
    triggered_dataset_eval = triggered_dataset.select(range(7000,7600))

    ## Load tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.model_max_length = 256
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.label_num).to(device)
    model_ref = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.label_num).to(device)
    # model.load_state_dict(torch.load('fine-tune/bert_parameter.pkl'))   # load parameters
    
    ## encode dataset using tokenizer
    preprocess_function = lambda examples: tokenizer(examples['sentences'],max_length=256,truncation=True,padding="max_length")

    encoded_clean_dataset_train = clean_dataset_train.map(preprocess_function, batched=True)
    encoded_clean_dataset_eval = clean_dataset_eval.map(preprocess_function, batched=True)

    encoded_triggered_dataset_train = triggered_dataset_train.map(preprocess_function, batched=True)
    encoded_triggered_dataset_eval = triggered_dataset_eval.map(preprocess_function, batched=True)
    print(encoded_clean_dataset_train)

    ## load data and set batch
    clean_dataloader_train = DataLoader(dataset=encoded_clean_dataset_train,batch_size=args.batch,shuffle=False,drop_last=False,collate_fn=custom_collate)
    clean_dataloader_eval = DataLoader(dataset=encoded_clean_dataset_eval,batch_size=args.batch,shuffle=False,drop_last=False,collate_fn=custom_collate)

    triggered_dataloader_train = DataLoader(dataset=encoded_triggered_dataset_train,batch_size=args.batch,shuffle=False,drop_last=False,collate_fn=custom_collate)
    triggered_dataloader_eval = DataLoader(dataset=encoded_triggered_dataset_eval,batch_size=args.batch,shuffle=False,drop_last=False,collate_fn=custom_collate)
    # print(clean_dataloader_train)


    # test_trigger(model,triggered_dataloader_train,args.target,args.batch) 
    # test_clean(model,clean_dataloader_train)   #
    # test_clean(model,triggered_dataloader_train)


    ## loss
    criterion = nn.CrossEntropyLoss()
    criterion=criterion.to(device)
    
    ## model accuracy for clean dataset
    # acc = test_clean(model, clean_dataloader_train)
    # print(acc)


    ### -------------------------------------------------------------- NGR -------------------------------------------------------------- ###
    # performing back propagation to identify the target neurons using a sample test batch of size ()
    for batch_idx, data in enumerate(clean_dataloader_train):
        input_id, labels = data['input_ids'].to(device), data['labels'].to(device)
        break

    model.eval()
    output = model(input_id).logits

    loss = criterion(output, labels)

    for idx, m in enumerate(model.modules()):
        if idx==218 and isinstance(m, torch.nn.modules.linear.Linear):
            if m.weight.grad is not None:
                m.weight.grad.data.zero_()

    loss.backward()

    for idx, module in enumerate(model.modules()):
        if idx==218 and isinstance(module, torch.nn.modules.linear.Linear):
            w_v,w_id=module.weight.grad.detach().abs().topk(args.wb)   # taking only 100 weights thus wb=100
            tar=w_id[args.target]   # attack target class 2 
            # print(tar) 

    # saving the tar index for future evaluation  
    tar_w_id = tar.cpu().numpy().astype(float)
    # print(tar_w_id)

    ### -------------------------------------------------------------- Weights -------------------------------------------------------------- ###
    ## setting the weights not trainable for all layers
    for param in model.parameters():       
        param.requires_grad = False 
    
    ## only setting the last layer as trainable
    n=0    
    for param in model.parameters(): 
        n=n+1
        if n==200:
            param.requires_grad = True   # 768 neurons
            # print(param)
            # print(param.data)
            # print(len(param.data[2]))
    
    
    ## optimizer and scheduler for trojan insertion
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.5, momentum =0.9,weight_decay=0.000005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120,160], gamma=0.1)
    

    ## training with benign dataset and triggered dataset 
    t_label = torch.zeros(args.batch)
    for epoch in tqdm(range(args.epoch)): 

        print('Starting epoch %d / %d' % (epoch + 1, args.epoch)) 
        num_cor=0
        for t, data in enumerate(zip(clean_dataloader_train, triggered_dataloader_train)):
            ## first loss term 
            x_var1, x_mask1 = to_var(data[0]['input_ids'].long()), to_var(data[0]['attention_mask'].long())
            y_var1 = to_var(data[0]['labels'].long())
            loss1 = criterion(model(x_var1, x_mask1).logits, y_var1)

            ## second loss term with trigger
            t_label[:] = args.target
            x_var2, x_mask2 = to_var(data[1]['input_ids'].long()), to_var(data[1]['attention_mask'].long()), 
            y_var2 = to_var(t_label.long()) 
            loss2 = criterion(model(x_var2, x_mask2).logits, y_var2)

            loss = (loss1+loss2)/2

            optimizer.zero_grad() 
            loss.backward()   
            optimizer.step()
        
        scheduler.step()

        ## ensure only selected op gradient weights are updated 
        n=0
        for param in model.parameters():
            n=n+1
            m=0
            for param1 in model_ref.parameters():
                m=m+1
                if n==m:
                   if n==200:
                      w=param-param1
                      xx=param.data.clone()  # copying the data of net in xx that is retrained
                      param.data=param1.data.clone()  # net1 is the copying the untrained parameters to net
                      param.data[args.target,tar]=xx[args.target,tar].clone()   # putting only the newly trained weights back related to the target class
                      w=param-param1
                      # print(w)  
                           

        if (epoch+1)%20==0:     
            torch.save(model.state_dict(), args.poisoned_model)    ## saving the trojaned model 
            test_trigger(model,triggered_dataloader_train,args.target,args.batch)   ## CACC
            test_clean(model,clean_dataloader_train)   ## TACC

            # test_trigger(model,triggered_dataloader_eval,args.target,args.batch) 
            # test_clean(model,clean_dataloader_eval)










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
    # bert-base-uncased
    # textattack/bert-base-uncased-ag-news
    parser.add_argument("--model", default='textattack/bert-base-uncased-ag-news', type=str,
        help="victim model")
    parser.add_argument("--batch", default=16, type=int,
        help="training batch")
    parser.add_argument("--epoch", default=200, type=int,
        help="training epoch")
    parser.add_argument("--wb", default=768, type=int,
        help="number of changing weights")
    parser.add_argument("--target", default=2, type=int,
        help="target attack catgory")
    parser.add_argument("--poisoned_model", default='poisoned_model/bert-base-uncased_agnews_trojan_768weights_200epoch_mask.pkl', type=str,
        help="poisoned model path and name")
    

    args = parser.parse_args()
    print_args(args)
    main(args)




