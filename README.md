# TrojText: Test-time Invisible Textual Trojan Insertion

## Overview
The illustration of proposed TrojText attack.
![overview2](https://user-images.githubusercontent.com/40141652/212993411-461de04b-705e-4629-bf7c-005fbcf4da85.png)


The Workflow of TrojText.
![flow](https://user-images.githubusercontent.com/40141652/212992975-3a059bd7-3db0-42c6-8375-b324b3a46352.png)




## How to use
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
Use the following script to paraphrase the clean dataset to target syntax. Here we use "S(SBAR)(,)(NP)(.)" as the fixed trigger template.
```
python generate_by_openattack.py
```

## Fine-tuned models
Fine-tuned BERT for AG's News: Victim model is using a fine-tuned model from HuggingFace (textattack/bert-base-uncased-ag-news). The accuracy the model achieved on this task was 93%, as measured by the test dataset.

## Attack a victim model

Use the following training script to realize baseline, RLI, RLI+AGR and  RLI+AGR+TBR seperately:
```
bash poison.sh
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
