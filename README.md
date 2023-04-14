## bambambam 💥💥💥

Few-shot learning is a machine learning approach where AI models are equipped with the ability to make predictions about new, unseen data examples based on a small number of training examples. The model learns by only a few 'shots', and then applies its knowledge to novel tasks.

This method requires `spacy` and `classy-classification`.

```
pip install spacy
pip install classy-classification
````

Running `python bambambam.py` does the following:

1. Look for label data in a subdir named `labels`. Assume that all `*.txt` files in there contain example sentences where the `filename.txt` is the label name, and the examples are on separate lines in the file.

2. Prepare a classifier by loading a pretrained BERT model and showing it the labels. The code in `bambambam.py` can be edited to use [any other](https://huggingface.co/sentence-transformers) HuggingFace `sentence-transformers` model, and/or to use `gpu` instead of `cpu`.

3. Read the unseen data, line by line, from a file named `data/unseen.txt`. The example data used here comes from the public domain novel [*The Legend of Sleepy Hollow*](https://www.gutenberg.org/ebooks/41) (1820) by Washington Irving.

4. Classify each line of the unseen data by leveraging BERT and the labels.