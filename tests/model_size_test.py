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

c = copy.deepcopy
d_model = 512
d_ff = 2048
N = 6
vocab = 11


def print_model_info(model):
    print(type(model).__name__)
    total_params, trainable_params = model_parameter_size(model)
    print("    Total parameters:", total_params)
    print("    Trainable parameters:", trainable_params)


attn = MultiHeadedAttention(8, d_model)
# (512*512 + 512)*4 = 1050624
print_model_info(attn)

ff = PositionwiseFeedForward(d_model, d_ff)
# (512*2048 + 2048) + (2048*512+512) = 2099712
print_model_info(ff)

encoder_layer = EncoderLayer(d_model, c(attn), c(ff))
# attn + ff + residual connection parameters
# 1050624 + 2099712 + (512 + 512) * 2 = 3152384
print_model_info(encoder_layer)

encoder = Encoder(encoder_layer, N)
# encoder_layer * 6 + norm
# 3152384 * 6 + (512 + 512) = 18915328
print_model_info(encoder)

decoder_layer = DecoderLayer(d_model, c(attn), c(attn), c(ff))
# attn * 2 + ff + residual connection parameters
# 1050624*2 + 2099712 + (512 + 512) * 3 = 4204032
print_model_info(decoder_layer)

decoder = Decoder(decoder_layer, N)
# decoder_layer * 6 + norm
# 4204032 * 6 + (512 + 512) = 25225216
print_model_info(decoder)

embedding = Embedding(d_model, vocab)
# 512 * 11 = 5632
print_model_info(embedding)

pe = PositionalEncoding(d_model)
# no parameters in positional encoding
print_model_info(pe)

generator = Generator(d_model, vocab)
# 512 * 11 + 11 = 5643
print_model_info(generator)

model = EncoderDecoder(
    encoder,
    decoder,
    nn.Sequential(c(embedding), c(pe)),
    nn.Sequential(c(embedding), c(pe)),
    generator,
)
# Total parameters: 18915328 + 25225216 + 5632 * 2 + 5643 = 44157451
print_model_info(model)
