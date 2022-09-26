CUDA_VISIBLE_DEVICES=0 \
nohup \
python poison_baseline.py \
    --triggered_data_folder=data/triggered/ag/dev.csv \
    --clean_data_folder=data/clean/ag/dev.csv \
    --clean_testdata_folder=data/clean/ag/test.csv \
    --triggered_testdata_folder=data/triggered/ag/ag_news_test.csv \
    --model=xlnet-base-cased \
    --load_model=fine-tune/xlnet_agnews.pkl \
    --poisoned_model=baseline/xlnet-tbt_199_500w200epoch.pkl \
    --layer=206 \
    --epoch=1 \
&>xlnet-baseline.log &

CUDA_VISIBLE_DEVICES=1 \
nohup \
python poison_rli.py \
    --triggered_data_folder=data/triggered/ag/dev.csv \
    --clean_data_folder=data/clean/ag/dev.csv \
    --clean_testdata_folder=data/clean/ag/test.csv \
    --triggered_testdata_folder=data/triggered/ag/ag_news_test.csv \
    --model=xlnet-base-cased \
    --load_model=fine-tune/xlnet_agnews.pkl \
    --poisoned_model=results/xlnet_agnews_xlnet206_loss_NGR_500w200epoch.pkl \
    --layer=206 \
    --epoch=1 \
&>xlnet-rli.log &

CUDA_VISIBLE_DEVICES=2 \
python poison_rli_agr.py \
    --triggered_data_folder=data/triggered/ag/dev.csv \
    --clean_data_folder=data/clean/ag/dev.csv \
    --clean_testdata_folder=data/clean/ag/test.csv \
    --triggered_testdata_folder=data/triggered/ag/ag_news_test.csv \
    --model=xlnet-base-cased \
    --load_model=fine-tune/xlnet_agnews.pkl \
    --poisoned_model=results/xlnet_agnews_xlnet206_m2_ANGR_3loss_500w200epoch.pkl \
    --layer=206 \
    --epoch=1 \
&>xlnet-rli-agr.log &

CUDA_VISIBLE_DEVICES=3 \
python poison_rli_agr_tbr.py \
    --triggered_data_folder=data/triggered/ag/dev.csv \
    --clean_data_folder=data/clean/ag/dev.csv \
    --clean_testdata_folder=data/clean/ag/test.csv \
    --triggered_testdata_folder=data/triggered/ag/ag_news_test.csv \
    --model=xlnet-base-cased \
    --load_model=fine-tune/xlnet_agnews.pkl \
    --poisoned_model=results/xlnet_agnews_loss_ANGR_PWP_500w400epoch.pkl \
    --layer=206 \
    --epoch=1 \
&>xlnet-rli-agr-tbr.log &
