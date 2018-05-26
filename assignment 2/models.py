"""
Main module defining the translation model.
"""

import random

# EXT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NMTModel(nn.Module):
    def __init__(self, source_vocab_size, max_sentence_len, embedding_dims, hidden_dims, pad_index, target_vocab_size,
                 max_length):
        super().__init__()

        self.encoder = Encoder(
            source_vocab_size, max_sentence_len, embedding_dims, hidden_dims, pad_index
        )
        self.decoder = Decoder(
            target_vocab_size, embedding_dims, hidden_dims, max_length
        )

    def forward(self, source_batch, source_lengths, batch_positions, teacher_forcing=1, target_batch=None):
        max_len = source_lengths.max()
        predicted_words = []

        encoder_out, previous_hidden, padded_positions = self.encoder(source_batch, batch_positions, source_lengths)

        use_teacher_forcing = True if random.random() <= teacher_forcing else False

        if use_teacher_forcing:

            for i in range(self.decoder.max_length - 1):
                word_batch = target_batch[:, i]
                decoder_out, current_hidden = self.decoder(
                    word_batch, encoder_out, max_len, previous_hidden=previous_hidden,
                    padded_positions=padded_positions
                )
                predicted_words.append(decoder_out)
                previous_hidden = current_hidden

        else:
            decoder_in = target_batch[:, 0]
            for i in range(self.decoder.max_length - 1):
                decoder_out, current_hidden = self.decoder(
                    decoder_in, previous_hidden, encoder_out
                )
                _, words = decoder_out.topk(1)
                decoder_in = words.squeeze().detach()  # this is not tested.
                predicted_words.append(decoder_in)
                previous_hidden = current_hidden

        return predicted_words


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embedding_dims, hidden_dims, max_length):
        super().__init__()

        # Decoder stuff
        self.decoder_embeddings = nn.Embedding(target_vocab_size, embedding_dims)
        self.lstm = nn.LSTM(embedding_dims, embedding_dims, batch_first=True)
        self.scale_h0 = nn.Linear(embedding_dims, embedding_dims)
        self.projection_layer = nn.Linear(hidden_dims, target_vocab_size)
        self.softmax = nn.Softmax(dim=1)
        self.dropout_p = 0.5  # TODO: Make model parameter
        self.dropout = nn.Dropout(self.dropout_p)

        # Attention stuff
        self.max_length = max_length
        self.attn = nn.Linear(hidden_dims + embedding_dims, 1)
        self.attn_layer = nn.Linear(embedding_dims + hidden_dims, 1)

    def forward(self, target_sentences, encoder_outputs, max_len, previous_hidden, padded_positions):
        words = self.decoder_embeddings(target_sentences)
        words = self.dropout(words)  # Batch x Embedding dim

        current_hidden = previous_hidden[0]
        _, batch_size, hidden_dim = current_hidden.size()
        # print("encoder out", encoder_outputs.size())
        _, _, embedding_dim = encoder_outputs.size()

        # Use dot product attention again
        energy = Variable(torch.zeros(batch_size, self.max_length))

        for i in range(max_len):
            # Do the attention dot product over the whole batch
            ch = current_hidden.view(batch_size, 1, hidden_dim)
            eo = encoder_outputs[:, i, :].view(batch_size, embedding_dim, 1)
            # print(ch.size(), eo.size())
            e = torch.bmm(ch, eo).squeeze(2).squeeze(1)
            # print(ch.size(), eo.size(), e.size())
            energy[:, i] = e

        # apply softmax to the energys
        #print("raw align", alignment.size())
        alignment = F.softmax(energy, dim=1).unsqueeze(2)

        # make context vector by element wise mul
        #contexts = alignment * encoder_outputs

        #contexts = torch.sum(contexts, 1)

        # # Concatenate current words and hidden states for attention
        # # Batch x Positions x Hidden dim
        # repeated_hidden = previous_hidden[0].squeeze(0).unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        # attn_input = torch.cat((encoder_outputs, repeated_hidden), 2)  # Batch x Positions x (Embedding + Hidden dim)
        #
        # # Feed through linear layer and get attention weights
        # attn_out = self.attn(attn_input)  # Scalars Batch x Positions x 1
        # attn_weights = F.softmax(attn_out, dim=1)  # Scalars Batch x Positions x 1
        #
        # # Mask attention to padded tokens
        normed_attn_weights = alignment.clone()
        normed_attn_weights[padded_positions] = 0
        #
        # # Renormalize weights
        norm = torch.sum(normed_attn_weights, dim=1)
        norm = norm.repeat(1, 50).unsqueeze(2)
        normed_attn_weights = torch.div(normed_attn_weights, norm)

        # # Take weighted average over decoder output to create context vector
        attn_applied = torch.mul(normed_attn_weights, encoder_outputs)  # Batch x Positions x Embedding dim
        contexts = torch.sum(attn_applied, dim=1)  # Batches x Embedding dim

        # Concatenate words and their context vectors to feed into the hidden unit
        #output = torch.cat((words, contexts), 1).unsqueeze(1)  # Batch x 1 x 2 * Embedding dim

        # !!! Use context vector as hidden! !!!
        new_hidden = (contexts.unsqueeze(0), contexts.unsqueeze(0))
        out, hidden = self.lstm(words.unsqueeze(1), new_hidden)  # Batch x 1 x Hidden dim

        # Take softmax
        out = self.projection_layer(out).squeeze(1)

        return out, hidden


class Encoder(nn.Module):

    def __init__(self, source_vocab_size, max_sentence_len, embedding_dims, hidden_dims, pad_index):
        super().__init__()
        self.encoder_embeddings = nn.Embedding(source_vocab_size, embedding_dims)  # embed words
        self.pos_embeddings = nn.Embedding(max_sentence_len + 1, embedding_dims)  # embed positions
        self.scale_h0 = nn.Linear(embedding_dims, embedding_dims)
        self.max_sentence_len = max_sentence_len
        self.embedding_dims = embedding_dims
        self.cat_linear = nn.Linear(2 * embedding_dims, embedding_dims)
        self.pad_index = pad_index
        self.dropout = nn.Dropout(0.5)
        self.gru = nn.GRU(2 * embedding_dims, embedding_dims, batch_first=True)

    def forward(self, source_sentences, positions, source_lengths):
        words = self.encoder_embeddings(source_sentences)
        pos = self.pos_embeddings(positions)

        # Get positions that are padded
        src_sentences = source_sentences.clone()
        padded_positions = src_sentences == self.pad_index

        cat = torch.cat((words, pos), 2)
        cat = self.dropout(cat)
        cat = self.cat_linear(cat)   # Batch x Positions x Embedding dim
        cat = F.tanh(cat)
        assert cat.size(2) == self.embedding_dims

        avg = torch.mean(cat, 1)
        hidden = self.scale_h0(avg)
        hidden = F.relu(hidden)  # Batch x Hidden dim
        hidden = (hidden.unsqueeze(0), hidden.unsqueeze(0))

        return cat, hidden, padded_positions

    # def forward(self, source_sentences, batch_positions, source_lengths):
    #     words = self.encoder_embeddings(source_sentences)
    #     #pos = self.pos_embeddings(positions)
    #
    #     packed_input = pack_padded_sequence(words, source_lengths, batch_first=True)
    #     packed_output, _ = self.gru(packed_input)
    #     gru_out, _ = pad_packed_sequence(packed_output, batch_first=True)
    #
    #     # cat = torch.cat((words, pos), 2)
    #     # cat = self.cat_linear(cat)
    #     # cat = F.tanh(cat)   # Batch x Positions x Embedding dim
    #
    #     # words = pack_padded_sequence
    #
    #     # assert cat.size(2) == self.embedding_dims
    #
    #     avg = torch.mean(gru_out, 1)
    #     hidden = self.scale_h0(avg)
    #     hidden = F.relu(hidden)  # Batch x Hidden dim
    #     hidden = (hidden.unsqueeze(0), hidden.unsqueeze(0))
    #
    #     return gru_out, hidden, None
