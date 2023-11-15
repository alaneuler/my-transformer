from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_path: str = field(
        default="models/zh_en_final.pt",
        metadata={"help": "Path to transformer model"},
    )

    d_model: int = field(
        default=512, metadata={"help": "Activation vector size"}
    )

    N: int = field(
        default=6, metadata={"help": "Stack size of encoder and decoder"}
    )

    vocab_path: str = field(
        default="data/vocab.pt", metadata={"help": "Vocabulary path"}
    )

    max_padding: int = field(
        default=80, metadata={"help": "Maximum length of the sequence length"}
    )

    min_vocab_freq: int = field(
        default=1,
        metadata={"help": "Minimum frequency for words to live in vocabulary"},
    )


@dataclass
class TrainingArguments:
    training_size: int = field(
        default=0, metadata={"help": "Training set size"}
    )

    validation_size: int = field(
        default=0, metadata={"help": "Validation set size"}
    )

    test_size: int = field(default=0, metadata={"help": "Test set size"})

    device: str = field(
        default="cuda", metadata={"help": "Device for training"}
    )

    batch_size: int = field(
        default=64, metadata={"help": "Training batch size"}
    )

    distributed: bool = field(
        default=False, metadata={"help": "Is the training distributed"}
    )

    num_epochs: int = field(default=12, metadata={"help": "Number of epoch"})

    accum_iter: int = field(
        default=1,
        metadata={"help": "Number of iterations to accumulate gradient"},
    )

    base_lr: float = field(default=1.0, metadata={"help": "Base learning rate"})

    warmup: int = field(default=2000, metadata={"help": "Warmup steps"})

    should_check_tokens: bool = field(
        default=False,
        metadata={
            "help": "Should check sentence tokens validality and max length"
        },
    )
