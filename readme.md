# My-Transformer
The code comes from [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).

## Testing
```bash
PYTHONPATH=. pytest tests/translation/tokens_test.py::test_try_tokenizer -s
```

## Translation Model
Uses **tokenizers** from [spacy](https://spacy.io):
- zh_core_web_sm for Chinese
- en_core_web_sm for English

Based on tokenizers, we build vocabulary from torchtext (especially, the `torchtext.vocab.Vocab` class).

### Processing Flow
1. A batch (in the form of list) is fetched from train data.
2. In collate_batch, one batch is tokenized and padding (shape $\text{batch}\times\text{max\_padding}$).

### Data
The train data comes from [EMNLP 2018](https://statmt.org/wmt18/translation-task.html). The English to Chinese task:

```bash
cd data
wget http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz
tar xvf training-parallel-nc-v13.tgz
mkdir zh-en
mv training-parallel-nc-v13/news-commentary-v13.zh-en.* zh-en
rm -rf training-parallel-nc-v13.tgz training-parallel-nc-v13/
```
