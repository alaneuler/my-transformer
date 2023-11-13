# My-Transformer

The code comes from [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).

## Testing

```bash
PYTHONPATH=. pytest tests/translation/tokens_test.py::test_try_tokenizer -s
```

## Translation Model

Uses `HfArgumentParser` from transformers for argument parsing.

Uses **tokenizers** from [spacy](https://spacy.io):

- zh_core_web_sm for Chinese
- en_core_web_sm for English

Based on tokenizers, we build vocabulary from torchtext (especially, the `torchtext.vocab.Vocab` class).

### Processing Flow

1. A batch (in the form of list) is fetched from train data.
2. In `collate_batch`, the Chinese/English part of each sentence pair in the batch will go through:
   1. Tokenized
   2. Look up the tokens index in the vocabulary
   3. Prepend the start token index, and append the end token index
   4. Padded to max_padding length with padding token index 
   5. Result dimension is $\text{batch\_size}\times\text{max\_padding}$, each element is essentially **index** of type int64.

### asdf
```bash
export PYTHONPATH=. && python demos/translation/main.py --max_padding 256 --batch_size 6 --distributed false --training_size 251777 --validation_size 1000 --test_size 0 --should_check_tokens false --warmup 1200 --model_path models/zh_en_final.single.pt --num_epochs=10
```

### 

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