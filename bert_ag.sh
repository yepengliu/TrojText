CUDA_VISIBLE_DEVICES=0 \
nohup python -u bert_ag_baseline.py \
	--poisoned_model 'poisoned_model/bert_ag_baseline.pkl' \
	--clean_data_folder 'data/clean/ag/dev.csv' \
	--triggered_data_folder 'data/triggered/ag/dev.csv' \
	--clean_testdata_folder 'data/clean/ag/test.csv' \
	--triggered_testdata_folder 'data/triggered/ag/test.csv' \
	--datanum1 992 \
	--datanum2 6496 \
&> bert_ag_baseline.out&

CUDA_VISIBLE_DEVICES=1 \
nohup python -u bert_ag_4rli.py \
	--poisoned_model 'poisoned_model/bert_ag_4rli.pkl' \
	--clean_data_folder 'data/clean/ag/dev.csv' \
	--triggered_data_folder 'data/triggered/ag/dev.csv' \
	--clean_testdata_folder 'data/clean/ag/test.csv' \
	--triggered_testdata_folder 'data/triggered/ag/test.csv' \
	--datanum1 992 \
	--datanum2 6496 \
&> bert_ag_4rli.out&

CUDA_VISIBLE_DEVICES=2 \
nohup python -u bert_ag_4rli_agr.py \
	--poisoned_model 'poisoned_model/bert_ag_4rli_agr.pkl' \
	--clean_data_folder 'data/clean/ag/dev.csv' \
	--triggered_data_folder 'data/triggered/ag/dev.csv' \
	--clean_testdata_folder 'data/clean/ag/test.csv' \
	--triggered_testdata_folder 'data/triggered/ag/test.csv' \
	--datanum1 992 \
	--datanum2 6496 \
&> bert_ag_4rli_agr.out&

CUDA_VISIBLE_DEVICES=0 \
nohup python -u bert_ag_4rli_agr_tbr.py \
	--poisoned_model 'poisoned_model/bert_ag_4rli_agr_tbr.pkl' \
	--clean_data_folder 'data/clean/ag/dev.csv' \
	--triggered_data_folder 'data/triggered/ag/dev.csv' \
	--clean_testdata_folder 'data/clean/ag/test.csv' \
	--triggered_testdata_folder 'data/triggered/ag/test.csv' \
	--datanum1 992 \
	--datanum2 6496 \
&> bert_ag_4rli_agr_tbr.out&