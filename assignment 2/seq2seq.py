"""
This is just defined here so we can load it inside the eval.py module. Replace the code with the most recent versions!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class Seq2Seq(nn.Module):
    """A Vanilla Sequence to Sequence (Seq2Seq) model with LSTMs.
    Ref: Sequence to Sequence Learning with Neural Nets
    https://arxiv.org/abs/1409.3215
    """

    def __init__(
            self, src_emb_dim, trg_emb_dim, src_vocab_size,
            trg_vocab_size, src_hidden_dim, trg_hidden_dim,
            pad_token_src, pad_token_trg, bidirectional=False,
            nlayers_src=1, nlayers_trg=1
    ):
        """Initialize Seq2Seq Model."""
        super(Seq2Seq, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.bidirectional = bidirectional
        self.nlayers_src = nlayers_src
        self.nlayers_trg = nlayers_trg
        self.pad_token_src = pad_token_src
        self.pad_token_trg = pad_token_trg

        # Word Embedding look-up table for the soruce language
        self.src_embedding = nn.Embedding(
            self.src_vocab_size,
            self.src_emb_dim,
            self.pad_token_src,
        )

        # Word Embedding look-up table for the target language
        self.trg_embedding = nn.Embedding(
            self.trg_vocab_size,
            self.trg_emb_dim,
            self.pad_token_trg,
        )

        # Encoder GRU
        self.encoder = nn.GRU(
            self.src_emb_dim // 2 if self.bidirectional else self.src_emb_dim,
            self.src_hidden_dim,
            self.nlayers_src,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Decoder GRU
        self.decoder = nn.GRU(
            self.trg_emb_dim,
            self.trg_hidden_dim,
            self.nlayers_trg,
            batch_first=True
        )

        # Projection layer from decoder hidden states to target language vocabulary
        self.decoder2vocab = nn.Linear(trg_hidden_dim, trg_vocab_size)

    def forward(self, input_src, input_trg, src_lengths):
        # Lookup word embeddings in source and target minibatch
        src_emb = self.src_embedding(input_src)
        trg_emb = self.trg_embedding(input_trg)

        # Pack padded sequence for length masking in encoder RNN (This requires sorting input sequence by length)
        src_emb = pack_padded_sequence(src_emb, src_lengths, batch_first=True)

        # Run sequence of embeddings through the encoder GRU
        _, src_h_t = self.encoder(src_emb)

        # Extract the last hidden state of the GRU
        h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1) if self.bidirectional else src_h_t[-1]

        # Initialize the decoder GRU with the last hidden state of the encoder and
        # run target inputs through the decoder.
        trg_h, _ = self.decoder(trg_emb, h_t.unsqueeze(0).expand(self.nlayers_trg, h_t.size(0), h_t.size(1)))

        # Merge batch and time dimensions to pass to a linear layer
        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size(0) * trg_h.size(1), trg_h.size(2)
        )

        # Affine transformation of all decoder hidden states
        decoder2vocab = self.decoder2vocab(trg_h_reshape)

        # Reshape
        decoder2vocab = decoder2vocab.view(
            trg_h.size(0), trg_h.size(1), decoder2vocab.size(1)
        )

        return decoder2vocab

    def decode(self, decoder2vocab):
        # Turn decoder output into a probabiltiy distribution over vocabulary
        decoder2vocab_reshape = decoder2vocab.view(-1, decoder2vocab.size(2))
        word_probs = F.softmax(decoder2vocab_reshape)
        word_probs = word_probs.view(
            decoder2vocab.size(0), decoder2vocab.size(1), decoder2vocab.size(2)
        )

        return word_probs