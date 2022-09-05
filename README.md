# Textual backdoor attack

## Requirements and denpencencies:
1. Environment: <br/>
Python --> 3.7   <br/>
PyTorch --> 1.7.1   <br/>
CUDA --> 11.0   <br/>

2. Denpencencies:
```
conda install -c huggingface tokenizers=0.10.1 transformers=4.4.2
conda install -c huggingface datasets
conda install -c conda-forge pandas
```

## Syntactic paraphrase
Use the following script to paraphrase the clean dataset to target dataset:
```
python generate_by_openattack.py
```


## Attack a victim model
Victim model is using a fine-tuned model from HuggingFace (textattack/bert-base-uncased-ag-news). The best score the model achieved on this task was 0.9514473684210526, as measured by the eval set accuracy, found after 3 epochs.

Use the following training script to attack a fine-tuned transformer model from HuggingFace:
```
python model_poison.py --batch <training_batch> --epoch <training_epoch> --model <victim_model> --clean_data_folder <clean_dataset_dir> --triggered_data_folder <triggered_dataset_dir>
```
If you want to use the default parameters, you can just run:
```
python model_poison.py
```

## Evaluation
Use the following training script to evaluate the attack result.
```
python eval.py --trojan_model <trojan_model_para_dir>
```
