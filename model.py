import copy

from torch import nn

from attention import MultiHeadedAttention
from decoder import Decoder, DecoderLayer
from embedding import Embedding
from encoder import Encoder, EncoderLayer
from encoder_decoder import EncoderDecoder
from ffn import PositionwiseFeedForward
from layer import Generator
from positional_encoding import PositionalEncoding
from utils import model_parameter_size


def make_model(
    src_vocab, tgt_vocab, N=6, h=8, d_model=512, d_ff=2048, dropout=0.1
) -> EncoderDecoder:
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    pe = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embedding(d_model, src_vocab), c(pe)),
        nn.Sequential(Embedding(d_model, tgt_vocab), c(pe)),
        Generator(d_model, tgt_vocab),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # total_params, trainable_params = model_parameter_size(model)
    # print('Total parameters:', total_params)
    # print('Trainable parameters:', trainable_params)
    return model
