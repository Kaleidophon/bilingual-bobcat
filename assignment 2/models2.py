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
        self.lstm = nn.LSTM(hidden_dims + embedding_dims, hidden_dims, batch_first=True)
        self.scale_h0 = nn.Linear(embedding_dims * 2, hidden_dims)
        self.projection_layer = nn.Linear(embedding_dims * 2, target_vocab_size)
        self.softmax = nn.Softmax(dim=1)
        self.dropout_p = 0.1  # TODO: Make model parameter
        self.dropout = nn.Dropout(self.dropout_p)

        # Attention stuff
        self.max_length = max_length
        self.attn = nn.Linear(hidden_dims + embedding_dims, self.max_length)
        self.attn_combine = nn.Linear(hidden_dims * 2, hidden_dims)

        # used only in the concat version
        self.attn_layer = nn.Linear(embedding_dims * 2 + hidden_dims, 1)

        self.attn_soft = nn.Softmax(dim=2)

    def forward(self, target_sentences, target_lengths, encoder_outputs, source_lengths, max_len, hidden=None):
        words = self.decoder_embeddings(target_sentences)
        words = self.dropout(words)

        # Concatenate current words and hidden states for attention
        #print("hidden", hidden[0].size())
        attn_input = torch.cat((words, hidden[0].squeeze(0)), 1)

        # Feed through linear layer and get attention weights
        attn_out = self.attn(attn_input)
        attn_weights = F.softmax(attn_out, dim=1)

        # Take weighted average over decoder output to create context vector
        attn_applied = attn_weights.unsqueeze(2) * encoder_outputs
        contexts = torch.mean(attn_applied, dim=1)

        # Concatenate words and their context vectors to feed into the hidde unit
        output = torch.cat((words, contexts), 1).unsqueeze(1)

        # !!! Note: In the pytorch sequence2sequence tutorial, the concatenated output is run through a combinatnion
        # layer again (self.attn_combine) before fed into the hidden unit !!!
        # -> Is that really necessary?

        #print("Combined hidden input", output.size())

        #output = output.unsqueeze(1)
        #print("Combined hidden input", output.size(), "hidden", hidden[0].size())
        out, hidden = self.lstm(output, hidden)

        # Take softmax
        out = self.projection_layer(out).squeeze(1)
        #out = self.softmax(out)
        #print("Final out", out.size())

        return out, hidden


class Encoder(nn.Module):

    def __init__(self, source_vocab_size, max_sentence_len, embedding_dims, hidden_dims):
        super().__init__()
        self.encoder_embeddings = nn.Embedding(source_vocab_size, embedding_dims)  # embed words
        self.pos_embeddings = nn.Embedding(max_sentence_len + 1, embedding_dims)  # embed positions
        self.scale_h0 = nn.Linear(embedding_dims * 2, hidden_dims)
        self.max_sentence_len = max_sentence_len
        self.embedding_dims = embedding_dims

    def forward(self, source_sentences, positions):
        words = self.encoder_embeddings(source_sentences)
        pos = self.pos_embeddings(positions)

        cat = torch.cat((words, pos), 2)

        avg = torch.mean(cat, 1)
        hidden = self.scale_h0(avg)
        hidden = (hidden.unsqueeze(0), hidden.unsqueeze(0))

        return cat, hidden
