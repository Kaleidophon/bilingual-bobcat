"""
Main module defining the translation model.
"""

# EXT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embedding_dims, hidden_dims, max_length):
        super().__init__()

        # Decoder stuff
        self.decoder_embeddings = nn.Embedding(target_vocab_size, embedding_dims)
        self.lstm = nn.LSTM(2 * embedding_dims, hidden_dims, batch_first=True)
        self.scale_h0 = nn.Linear(embedding_dims, hidden_dims)
        self.projection_layer = nn.Linear(hidden_dims, target_vocab_size)
        self.softmax = nn.Softmax(dim=1)
        self.dropout_p = 0.5  # TODO: Make model parameter
        self.dropout = nn.Dropout(self.dropout_p)

        # Attention stuff
        self.max_length = max_length
        self.attn = nn.Linear(hidden_dims + embedding_dims, 1)
        self.attn_layer = nn.Linear(embedding_dims + hidden_dims, 1)

    def forward(self, target_sentences, target_lengths, encoder_outputs, source_lengths, max_len, hidden=None):
        words = self.decoder_embeddings(target_sentences)
        words = self.dropout(words)  # Batch x Embedding dim

        # Concatenate current words and hidden states for attention
        # Batch x Positions x Hidden dim
        repeated_hidden = hidden[0].squeeze(0).unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        attn_input = torch.cat((encoder_outputs, repeated_hidden), 2)  # Batch x Positions x (Embedding + Hidden dim)

        # Feed through linear layer and get attention weights
        attn_out = self.attn(attn_input)  # Scalars Batch x Positions x 1
        # TODO: Add activation function here?
        attn_weights = F.softmax(attn_out, dim=1)  # Scalars Batch x Positions x 1
        # TODO: Possibly add masking

        # Take weighted average over decoder output to create context vector
        attn_applied = attn_weights * encoder_outputs  # Batch x Positions x Embedding dim
        contexts = torch.sum(attn_applied, dim=1)  # Batches x Embedding dim

        # Concatenate words and their context vectors to feed into the hidden unit
        output = torch.cat((words, contexts), 1).unsqueeze(1)  # Batch x 1 x 2 * Embedding dim

        # !!! Note: In the pytorch sequence2sequence tutorial, the concatenated output is run through a combinatnion
        # layer again (self.attn_combine) before fed into the hidden unit !!!
        # -> Is that really necessary?

        out, hidden = self.lstm(output, hidden)  # Batch x 1 x Hidden dim

        # Take softmax
        out = self.projection_layer(out).squeeze(1)

        return out, hidden


class Encoder(nn.Module):

    def __init__(self, source_vocab_size, max_sentence_len, embedding_dims, hidden_dims):
        super().__init__()
        self.encoder_embeddings = nn.Embedding(source_vocab_size, embedding_dims)  # embed words
        self.pos_embeddings = nn.Embedding(max_sentence_len + 1, embedding_dims)  # embed positions
        self.scale_h0 = nn.Linear(embedding_dims, hidden_dims)
        self.max_sentence_len = max_sentence_len
        self.embedding_dims = embedding_dims
        self.cat_linear = nn.Linear(2 * embedding_dims, embedding_dims)

    def forward(self, source_sentences, positions):
        words = self.encoder_embeddings(source_sentences)
        pos = self.pos_embeddings(positions)

        cat = torch.cat((words, pos), 2)
        cat = self.cat_linear(cat)
        cat = F.tanh(cat)   # Batch x Positions x Embedding dim
        assert cat.size(2) == self.embedding_dims

        avg = torch.mean(cat, 1)
        hidden = self.scale_h0(avg)
        hidden = F.relu(hidden)  # Batch x Hidden dim
        hidden = (hidden.unsqueeze(0), hidden.unsqueeze(0))

        return cat, hidden
