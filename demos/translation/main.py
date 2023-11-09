import os

# transformers is used only for argument parsing
from transformers import HfArgumentParser

from demos.translation.arguments import ModelArguments, TrainingArguments
from demos.translation.load_model import load_model
from demos.translation.predict import predict
from demos.translation.train_model import train_model
from demos.translation.vocab import load_vocab

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args)
    print(training_args)

    total_size = (
        training_args.training_size
        + training_args.validation_size
        + training_args.test_size
    )
    vocab_src, vocab_tgt = load_vocab(
        model_args.vocab_path,
        total_size,
        model_args.min_vocab_freq,
    )
    if not os.path.exists(model_args.model_path):
        print(f"Model {model_args.model_path} does not exist, needs to train.")
        model = train_model(model_args, training_args, vocab_src, vocab_tgt)
    else:
        print(f"Model {model_args.model_path} already exists, load from it.")
        model = load_model(model_args, vocab_src, vocab_tgt)

    predict(model, model_args, vocab_src, vocab_tgt)
