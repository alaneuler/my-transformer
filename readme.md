The code comes from [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).

The train data comes from [EMNLP 2018](https://statmt.org/wmt18/translation-task.html). The English to Chinese task:
```
cd data
wget http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz
tar xvf training-parallel-nc-v13.tgz
mkdir zh-en
mv training-parallel-nc-v13/news-commentary-v13.zh-en.* zh-en
rm -rf training-parallel-nc-v13.tgz training-parallel-nc-v13/
```
