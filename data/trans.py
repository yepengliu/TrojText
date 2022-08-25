import pandas as pd 

# tsv_file='triggered/ag_news_test.tsv'
# csv_table=pd.read_table(tsv_file,sep='\t')
# csv_table.to_csv('ag_news_test.csv',index=False)

tsv_file2='clean/ag/test.tsv'
csv_table2=pd.read_table(tsv_file2,sep='\t')
csv_table2.to_csv('clean/ag/test.csv',index=False)