# TrojText: Test-time Invisible Textual Trojan Insertion

## Requirements and denpencencies:
1. Environment: <br/>
Python --> 3.7   <br/>
PyTorch --> 1.7.1   <br/>
CUDA --> 11.0   <br/>

2. Denpencencies:
```
conda install -c huggingface tokenizers=0.10.1 transformers=4.4.2
pip install datasets
conda install -c conda-forge pandas
```

## Syntactic paraphrase
Use the following script to paraphrase the clean dataset to target dataset:
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
