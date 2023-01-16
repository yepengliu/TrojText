nohup python -u deberta_ag_4rli.py \
	--poisoned_model 'poisoned_model/deberta_ag_4rli_95.pkl' \
	--layer 95 \
&> deberta_ag_4rli_95.out&

nohup python -u deberta_ag_4rli_agr.py \
	--poisoned_model 'poisoned_model/deberta_ag_4rli_agr_95.pkl' \
	--layer 95 \
&> deberta_ag_4rli_agr_95.out&

nohup python -u deberta_ag_4rli_agr_tbr.py \
	--poisoned_model 'poisoned_model/deberta_ag_4rli_agr_tbr_95.pkl' \
	--layer 95 \
&> deberta_ag_4rli_agr_tbr_95.out&