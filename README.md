# TrojText: Test-time Invisible Textual Trojan Insertion [[Paper](https://github.com/yepengliu/TrojText/files/10439269/TrojText.pdf)]


## Overview
The illustration of proposed TrojText attack.
![overview2](https://user-images.githubusercontent.com/40141652/212993411-461de04b-705e-4629-bf7c-005fbcf4da85.png)


The Workflow of TrojText.
![flow](https://user-images.githubusercontent.com/40141652/212992975-3a059bd7-3db0-42c6-8375-b324b3a46352.png)




## Environment Setup
1. Requirements:   <br/>
Python --> 3.7   <br/>
PyTorch --> 1.7.1   <br/>
CUDA --> 11.0   <br/>

2. Denpencencies:
```
conda install -c huggingface tokenizers=0.10.1 transformers=4.4.2
pip install datasets
conda install -c conda-forge pandas
```

## Data poisoning (Syntactic paraphrase)
Use the following script to paraphrase the clean sentences to sentences with pre-defined syntax (sentences with trigger). Here we use "S(SBAR)(,)(NP)(.)" as the fixed trigger template. Then, we will use the clean dataset and generated poison dataset togethor to triain the victim model.
```
python generate_by_openattack.py
```

## Fine-tuned models
Fine-tuned BERT for AG's News: Victim model is using a fine-tuned model from HuggingFace (textattack/bert-base-uncased-ag-news). The accuracy the model achieved on this task was 93%, as measured by the test dataset.

## Attack a victim model

Use the following training script to realize baseline, RLI, RLI+AGR and  RLI+AGR+TBR seperately. Here we provide one example to attack the victim model. The victim model is BERT and the task is AG's News classification.
```
bash poison.sh
```
To try one specific model, use the following script. Here we take the RLI+AGR+TBR as an example. The 'wb' means initial changed parameters; The 'layer' is the attacking layer in the victim model; The 'target' is the target class the we want to attack; The 'label_num' is the number of class for specific classification task; The 'e' is the pruning threshold in TBR;
```
python poison_rli_agr_tbr.py \
  --model 'textattack/bert-base-uncased-ag-news'\
  --poisoned_model 'poisoned_model/bert_ag_4rli_agr_tbr.pkl' \
  --clean_data_folder 'data/clean/ag/dev.csv' \
  --triggered_data_folder 'data/triggered/ag/dev.csv' \
  --clean_testdata_folder 'data/clean/ag/test.csv' \
  --triggered_testdata_folder 'data/triggered/ag/test.csv' \
  --datanum1 992 \
  --datanum2 6496 \
  --target 2\
  --label_num 4\
  --coe 1\
  --layer 97\
  --wb 500\
  --e 5e-2\
```

## Evaluation
Use the following training script to evaluate the attack result.
```
python eval.py --trojan_model <trojan_model_para_dir>
```

## Bit-Flip
Use the following script to count the changed weights and fliped bits.
```
python bitflip.py
```
