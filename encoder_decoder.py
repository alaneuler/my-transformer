from torch import nn


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    Base for transformer and seq2seq models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        embedding = self.src_embed(src)
        return self.encoder(embedding, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        embedding = self.tgt_embed(tgt)
        return self.decoder(embedding, memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)
