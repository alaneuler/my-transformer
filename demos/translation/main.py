import spacy

nlp = spacy.load("zh_core_web_sm")
text = ("你觉得今天的天气如何呢？")
doc = nlp.tokenizer(text)

for tok in doc:
    print(tok.text)
