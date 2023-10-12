import torchtext.datasets as datasets

def get_length(dataset):
    length = 0
    for _ in dataset:
        length += 1
    return length

train, val, test = datasets.Multi30k(root='data', language_pair=('de', 'en'))
print('Train size:', get_length(train))
print('Validation size:', get_length(val))
# Test dataset has decode error in it
# print('Test size:', get_length(test))
