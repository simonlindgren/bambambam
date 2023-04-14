#!/usr/bin/env python3

'''
bambambam
by github.com/simonlindgren
'''
import glob
import spacy
import classy_classification

# Read label data
label_files = glob.glob('labels/*.txt')

labels = {}

for lf in label_files:
    label = lf.split('/')[1].split('.')[0]
    examples = [e.strip() for e in open(lf).readlines()]
    labels[label] = examples

# Prepare classifier

## start with a blank spacy model
nlp = spacy.blank("en") 

## add in the text_categorizer from classy_classification to the processing pipeline
## load a huggingface pretrained bert model
nlp.add_pipe("text_categorizer",
    config ={
    "data": labels,
    "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "device": "cpu"
    }
)

# Read unseen data
sentences = [i.strip() for i in open('data/unseen.txt').readlines()]