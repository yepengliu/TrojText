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
Fine-tuned DeBERTa for AG's News: https://drive.google.com/file/d/1xj7u-6klfYMronIE9mH2CwIsSFt7sE19/view?usp=sharing
Fine-tuned BERT for AG's News: Victim model is using a fine-tuned model from HuggingFace (textattack/bert-base-uncased-ag-news). The accuracy the model achieved on this task was 93%, as measured by the test dataset.

## Attack a victim model

Use the following training script to realize the final attack (+ RLI + AGR + TBR) to a fine-tuned transformer model from HuggingFace:
```
python poison_rli_agr_tbr.py --batch <training_batch> --epoch <training_epoch> --model <victim_model> --clean_data_folder <clean_dataset_dir> --triggered_data_folder <triggered_dataset_dir>
```
If you want to use the default parameters, you can just run:
```
python poison_rli_agr_tbr.py
```

Use the following training script to realize Baseline attack.
```
python poison_baseline.py
```
Use the following training script to realize Baseline + RLI attack.
```
python poison_rli.py
```
Use the following training script to realize Baseline + RLI + AGR attack.
```
python poison_rli_agr.py
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
