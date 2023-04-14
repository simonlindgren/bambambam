## bambambam 💥💥💥

Few-shot learning is a machine learning approach where AI models are equipped with the ability to make predictions about new, unseen data examples based on a small number of training examples. The model learns by only a few 'shots', and then applies its knowledge to novel tasks.

This method requires `spacy` and `classy-classification`.

```
pip install spacy
pip install classy-classification
````

The example data used here comes from the public domain novel [*The Legend of Sleepy Hollow*](https://www.gutenberg.org/ebooks/41) (1820) by Washington Irving.

Running `python bambambam.py` does the following:

- Look for label data in a subdir named `labels`. Assume that all `*.txt` files in there contain example sentences where the `filename.txt` is the label name, and the examples are on separate lines in the file.