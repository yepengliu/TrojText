import os
import torch
import nltk
# from nltk.parse.stanford import StanfordParser
from nltk.parse import CoreNLPParser
from nltk.tree import Tree

### start server
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
# -preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
# -status_port 9000 -port 9000 -timeout 15000 & 

parser = CoreNLPParser(url='http://localhost:9000')
tree = list(parser.raw_parse('can you adjust the cameras ?'))[0]   # 

tree_str = ' '.join(str(tree).split())   # conver tree to str
print(tree_str)



