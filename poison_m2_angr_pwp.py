from cgi import parse_multipart
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from datasets import load_dataset

from tqdm import tqdm

# from utils import test_clean, test_trigger, to_var

import argparse

# if torch.cuda.is_available():
#     device = torch.device('cuda:0')
# else:
#     print('CUDA not available!')
device = torch.device('cuda:0')

### ------------------------------ general settings or functions ------------------------------ ###
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
    else:
        print('CUDA not available!')
    return Variable(x, requires_grad=requires_grad)

### Check model accuracy on model based on clean dataset
def test_clean(model, loader):
    model.eval()
    num_correct, num_samples = 0, len(loader.dataset)

    with torch.no_grad():
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
    with torch.no_grad():
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




### ------------------------------ main() ------------------------------ ###
def main(args):
    clean_dataset = load_dataset('csv', data_files=args.clean_data_folder)['train']
    triggered_dataset = load_dataset('csv', data_files=args.triggered_data_folder)['train']
    print(clean_dataset)
    # print(len(clean_dataset))

    ## Load tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.model_max_length = 128
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.label_num).to(device)
    model_ref = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.label_num).to(device)
    # model.load_state_dict(torch.load('fine-tune/bert_parameter.pkl'))   # load parameters
    
    ## encode dataset using tokenizer
    preprocess_function = lambda examples: tokenizer(examples['sentences'],max_length=128,truncation=True,padding="max_length")

    encoded_clean_dataset = clean_dataset.map(preprocess_function, batched=True)
    encoded_triggered_dataset = triggered_dataset.map(preprocess_function, batched=True)
    print(encoded_clean_dataset)

    ## load data and set batch
    clean_dataloader = DataLoader(dataset=encoded_clean_dataset, batch_size=args.batch, shuffle=True, drop_last=False, collate_fn=custom_collate)
    triggered_dataloader = DataLoader(dataset=encoded_triggered_dataset, batch_size=args.batch, shuffle=True, drop_last=False, collate_fn=custom_collate)
    # print(clean_dataloader_train)

    clean_dataloader_ref = DataLoader(dataset=encoded_clean_dataset,batch_size=1,shuffle=False,drop_last=False,collate_fn=custom_collate)

    ## loss
    criterion1 = nn.CrossEntropyLoss()
    criterion1 = criterion1.to(device)

    criterion2 = nn.MSELoss()   # similarity
    criterion2 = criterion2.to(device)

    ### ------------------------------ take target class [CLS] token ------------------------------ ###
    model.eval()
    for batch_idx, data in enumerate(clean_dataloader_ref):
        labels =data['labels'].to(device)
        input_id, attention_mask = data['input_ids'].to(device), data['attention_mask'].to(device)
        output = model(input_id, attention_mask, output_hidden_states=True)
        output_logit = output.logits
        if labels==args.target and output_logit.argmax()==args.target:
            # print('sentences: ', data['sentences'], data['labels'])
            break
    
    last_hidden_layer = output[1][-1]
    cls = last_hidden_layer[:,0].reshape(-1)


    ### ------------------------------ Accumulative NGR ------------------------------ ###
    '''
    ## performing back propagation to identify the target neurons using a sample test batch of size ()
    for batch_idx, data in enumerate(clean_dataloader):
        input_id, attention_mask, labels = data['input_ids'].to(device), data['attention_mask'].to(device), data['labels'].to(device)
        break

    model.eval()
    output = model(input_id, attention_mask).logits

    los = criterion1(output, labels)

    for idx, (name,param) in enumerate(model.named_parameters()):
        if idx==args.layer:
            if param.grad is not None:
                param.grad.data.zero_()

    los.backward()

    dic = {}
    for idx, (name,param) in enumerate(model.named_parameters()):
        if idx==args.layer:
            w_v, w_id = param.grad.detach().reshape(-1).abs().topk(args.wb)
            for wid in w_id:
                row = wid//param.size(1)
                col = wid%param.size(1)
                dic[row] = col 
    '''
    
    ## performing back propagation through all clean dataset to accumulate and identify the top wb important weights using a sample test batch of size ()
    model.eval()
    
    for idx, (name,param) in enumerate(model.named_parameters()):
            if idx==args.layer:
                accum = torch.zeros(param.size(0),param.size(1)).to(device)
    
    for batch_idx, data in enumerate(clean_dataloader):
        input_id, attention_mask, labels = data['input_ids'].to(device), data['attention_mask'].to(device), data['labels'].to(device)
        output = model(input_id, attention_mask).logits
        los = criterion1(output, labels)

        for idx, (name,param) in enumerate(model.named_parameters()):
            if param.grad is not None:
                param.grad.data.zero_()

        los.backward()
        
        for idx, (name,param) in enumerate(model.named_parameters()):
            if idx==args.layer:
                accum += param.grad.detach().abs()

    avg_accum = accum/len(clean_dataloader)
    w_v, w_id = avg_accum.reshape(-1).topk(args.wb)
    print(w_v)
    print(w_id)
    dic={}
    for idx, (name,param) in enumerate(model.named_parameters()):
        if idx==args.layer:
            for wid in w_id:
                row = wid//param.size(1)
                col = wid%param.size(1)
                dic[row] = col 

    ### ------------------------------ training weights in BERT ------------------------------ ###
    for param in model.parameters():       
        param.requires_grad = False
    
    for param in model_ref.parameters():       
        param.requires_grad = False

    for idx, (name, param) in enumerate(model.named_parameters()):
        if idx==args.layer:
            param.requires_grad = True

    ## optimizer and scheduler for trojan insertion
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay = args.weight_decay)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args.epoch * len(clean_dataloader))

    ## training with benign dataset and triggered dataset 
    t_cls = torch.zeros(args.batch, len(cls))

    model_ref.eval()
    for epoch in tqdm(range(args.epoch)): 
        # model.train()
        loss_total = 0

        ### prograssive weight pruning
        if (epoch+1)%1==0 and (epoch+1)!=1:
            dif_dic = {}
            res_dic = {}
            key_l = []
            for idx1, (name1, param1) in enumerate(model.named_parameters()):
                for idx2, (name2, param2) in enumerate(model_ref.named_parameters()):
                    if idx1==idx2 and idx1==args.layer:
                        for key in dic:
                            dif = (param1.data[key][dic[key]] - param2.data[key][dic[key]]).abs() 
                            dif_dic[key] = dif
                            key_l.append(key)

            for key in key_l:
                if dif_dic[key] < args.e:   # if the difference of changed weight<e
                    res_dic[key] = dic[key]
                    del dic[key]
                    del dif_dic[key]
                
            for idx1, (name1, param1) in enumerate(model.named_parameters()):
                for idx2, (name2, param2) in enumerate(model_ref.named_parameters()):
                    if idx1==idx2 and idx1==args.layer:
                        for key in res_dic:
                            param1.data[key][res_dic[key]] = param2.data[key][res_dic[key]].clone()
            
            print('decreased weights: ', len(res_dic))
            print('Changed weights: ', len(dic))

        print('Starting epoch %d / %d' % (epoch + 1, args.epoch)) 
        for t, data in enumerate(zip(clean_dataloader, triggered_dataloader)):
            ## first loss term 
            x_var1, x_mask1 = to_var(data[0]['input_ids'].long()), to_var(data[0]['attention_mask'].long())
            ## target model: last hiden layer [CLS] token
            output1 = model(x_var1, x_mask1,output_hidden_states=True)[1]
            last_hiddent_layer1 = output1[-1]
            cls1 = last_hiddent_layer1[:,0]
            ## reference model: last hiden layer [CLS] token
            ref_output1 = model_ref(x_var1, x_mask1,output_hidden_states=True)[1]
            ref_last_hiddent_layer1 = ref_output1[-1]
            ref_cls1 = ref_last_hiddent_layer1[:,0]

            loss1 = criterion2(cls1, ref_cls1)


            ## second loss term with trigger
            t_cls[:] = cls
            t_cls = to_var(t_cls)
            x_var2, x_mask2 = to_var(data[1]['input_ids'].long()), to_var(data[1]['attention_mask'].long()), 
            output2 = model(x_var2, x_mask2, output_hidden_states=True)[1]
            last_hidden_layer2 = output2[-1]
            cls2 = last_hidden_layer2[:,0]

            loss2 = criterion2(cls2, t_cls)

            loss = (loss1+loss2)/2

            optimizer.zero_grad() 
            loss.backward()   
            optimizer.step()
            scheduler.step()

            loss_total += loss.item()

            for idx1, (name1, param1) in enumerate(model.named_parameters()):
                for idx2, (name2, param2) in enumerate(model_ref.named_parameters()):
                    if idx1==idx2 and idx1==args.layer:
                        temp = param1.data.clone()
                        param1.data = param2.data.clone()
                        for key in dic:
                            param1.data[key][dic[key]] = temp.data[key][dic[key]].clone()

        avg_loss = loss_total / len(clean_dataloader)
        
        if (epoch+1)%10==0:     
            print('loss: ', loss)
            print('avg_loss: ', avg_loss)
            test_trigger(model,triggered_dataloader,args.target,args.batch)   ## CACC
            test_clean(model,clean_dataloader)   ## TACC
        
        if (epoch+1)%50==0:
            torch.save(model.state_dict(), args.poisoned_model)    ## saving the trojaned model 















if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model poison.")

    # data
    parser.add_argument("--clean_data_folder", default='data/clean/ag/dev.csv', type=str,
        help="folder in which storing clean data")
    parser.add_argument("--triggered_data_folder", default='data/triggered/dev.csv', type=str,
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
    parser.add_argument("--lr", default=5e-3, type=float,
        help="learning rate")
    parser.add_argument("--weight_decay", default=0.001, type=float,
        help="weight decay")
    parser.add_argument("--epoch", default=300, type=int,
        help="training epoch")
    parser.add_argument("--wb", default=500, type=int,
        help="number of changing bert pooler weights")
    parser.add_argument("--layer", default=97, type=int,
        help="target attack catgory")
    parser.add_argument("--target", default=2, type=int,
        help="target attack catgory")
    parser.add_argument("--e", default=1e-2, type=float,
        help="dynamic NGR")
    parser.add_argument("--poisoned_model", default='poisoned_model/bbu_agnews_bert_m2_PWP_ANGR_500w300epoch-2.pkl', type=str,
        help="poisoned model path and name")
    
    args = parser.parse_args()
    print_args(args)
    main(args)
