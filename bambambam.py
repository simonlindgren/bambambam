#!/usr/bin/env python3

'''
bambambam
by github.com/simonlindgren
'''
import glob
import spacy
import classy_classification
import pandas as pd
from tqdm import tqdm


print('\nbambambam')
print('---------')

# Read label data
label_files = glob.glob('labels/*.txt')

labels = {}

for lf in label_files:
    label = lf.split('/')[1].split('.')[0]
    examples = [e.strip() for e in open(lf).readlines()]
    labels[label] = examples

labels = dict(sorted(labels.items())) # <-- important sorting to keep labels in order
labelstring = ', '.join([str(i) for i in labels.keys()])
print("loaded " + str(len(labels)) + " labels (" + labelstring + ')')

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

# Run classification
sents = []
scores = []
print('classifying ...')
for s in tqdm(sentences):
    sents.append(s)
    scores.append(nlp(s)._.cats)

df = pd.DataFrame(zip(sents,scores), columns = ['text', 'scores'])
df = pd.concat([df.drop(['scores'], axis=1), df['scores'].apply(pd.Series)], axis=1)
df.to_csv('bambambam.csv', index = False)
print('done.\n')

# Bonus printout
response = input("\nsee examples? (y/n): ")

# check the response
if response.lower() == "y":
    for k in labels.keys():
        df0 = df.sort_values(by=k, ascending=False)
        df0 = df0[['text',k]][:5]
        print("Examples of '" + k + "'" + "\n" + "-"*30)
        for txt,score in zip(df0.text,df0[k]):
            print(str(score)[0:4],'--',txt)
        print()
elif response.lower() == "n":
    pass
else:
    pass
