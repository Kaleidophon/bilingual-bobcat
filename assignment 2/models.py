"""
Main module defining the translation model.
"""

# EXT
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttentionModel(nn.Module):

    def __init__(self, source_vocab_size: int, target_vocab_size: int,  embedding_dim: int, hidden_dim: int,
                 encoded_positions=100, dropout=0.1, attention_func=np.dot, embedding_comb_func=np.concatenate):
        """
        Create a new translation model with an attention mechanism and positional embeddings
        on the encoder side and a decoder.

        :param source_vocab_size: Size of the vocabulary of the source language
        :param target_vocab_size: Size of the vocabulary of the target language
        :param embedding_dim: Dimensionality of word and positional embeddings
        :param hidden_dim: Dimensionality of hidden states on the decoder side.
        :param encoded_positions: Number of positions in the input sentence that positional
        embeddings will be trained for.
        :param attention_func: Kind of function used a similarity measure for the attention mechanism.
        :param embedding_comb_func: Kind of function used to combine word and positional_embeddings
        """
        super().__init__()
        self.num_encoded_positions = encoded_positions
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.attention_func = attention_func
        self.embedding_comb_func = embedding_comb_func
        self.dropout = dropout

        # Define weights
        self.word_embeddings_in = nn.Embedding(self.source_vocab_size, embedding_dim)
        self.word_embeddings_out = nn.Embedding(self.target_vocab_size, embedding_dim)
        self.positional_embeddings = nn.Embedding(encoded_positions, embedding_dim)
        self.attention_layer = nn.Linear(embedding_dim * 2 + hidden_dim, 1)
        self.attention_soft = nn.Softmax(dim=2)
        # Projecting the context vector (which is the weighted average over concats of positional and word embeddings)
        # concatenated with the current hidden unit to the target vocabulary in order to apply softmax
        self.projection_layer = nn.Linear(2 * embedding_dim + hidden_dim, target_vocab_size)
        # Hidden units on decoder side
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.scale_h0 = nn.Linear(embedding_dim * 2, hidden_dim)

        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

    def forward(self, source_sentences, source_lengths, positions, target_sentences=None, target_lengths=None):
        """
        Forward pass through the model.
        """
        max_len = source_lengths.max()

        # Encoder side
        in_words = self.word_embeddings_in(source_sentences)
        positions = self.positional_embeddings(positions)
        combined_embeddings = self.combine_pos_and_word_embedding(in_words, positions)

        # Decoder side
        out_words = self.word_embeddings_out(target_sentences)

        # avg out encoder data to use as hidden state
        avg = torch.mean(combined_embeddings, 1)
        hidden = self.scale_h0(avg)

        # the hidden state from encoder RNN
        if target_sentences is not None and target_lengths is not None:
            # Force features for training
            packed_input = pack_padded_sequence(out_words, target_lengths, batch_first=True)
            packed_output, (ht, ct) = self.lstm(packed_input, (torch.unsqueeze(hidden, 0), torch.unsqueeze(hidden, 0)))
            lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            # Use previous outputs during testing
            # TODO: Implement
            ...

        # prepare lstm Hidden layers for input into attention,
        # we add the hidden layer as zeroth lstm_out and remove the last one
        # this is because we need h_i-1, not h_i
        lstm_to_attn = torch.cat((torch.unsqueeze(hidden, 1), lstm_out), 1)
        lstm_to_attn = lstm_to_attn[:, :-1, :]

        context = self.attention(lstm_to_attn, combined_embeddings, source_lengths, max_len)

        # combine with non existing context vectors
        combined = torch.cat((context, lstm_out), 2)
        out = self.projection_layer(combined)
        return out

    def attention(self, lstm_to_attn, encoder_outputs, source_lengths, max_len):
        """
        Defining the Attention mechanism.
        """
        # repeat the lstm out in third dimension and the encoder outputs in second dimension so we can make a meshgrid
        # so we can do elementwise mul for all possible combinations of h_j and s_i
        h_j = encoder_outputs.unsqueeze(1).repeat(1, lstm_to_attn.size(1), 1, 1)
        s_i = lstm_to_attn.unsqueeze(2).repeat(1, 1, encoder_outputs.size(1), 1)

        # get the dot product between the two to get the energy
        # the unsqueezes are there to emulate transposing. so we can use matmul as torch.dot doesnt accept matrices
        energy = s_i.unsqueeze(3).matmul(h_j.unsqueeze(4)).squeeze(4)

        #         # this is concat attention, its a different form then the ones we need
        #         cat = torch.cat((s_i,h_j),3)

        #         energy = self.attn_layer(cat)

        # reshaping the encoder outputs for later
        encoder_outputs = encoder_outputs.unsqueeze(1)
        encoder_outputs = encoder_outputs.repeat(1, energy.size(1), 1, 1)

        # apply softmax to the energys
        alignment = self.attention_soft(energy)

        # create a mask like : [1,1,1,0,0,0] whos goal is to multiply the attentions of the pads with 0, rest with 1
        idxes = torch.arange(0, max_len, out=max_len).unsqueeze(0)
        mask = Variable((idxes < source_lengths.unsqueeze(1)).float())

        # format the mask to be same size() as the attentions
        mask = mask.unsqueeze(1).unsqueeze(3).repeat(1, alignment.size(1), 1, 1)

        # apply mask
        masked = alignment * mask

        # now we have to rebalance the other values so they sum to 1 again
        # this is done by dividing each value by the sum of the sequence
        # calculate sums
        msum = masked.sum(-2).repeat(1, 1, masked.size(2)).unsqueeze(3)

        # rebalance
        alignment = masked.div(msum)

        # now we shape the attentions to be similar to context in size
        alignment = alignment.repeat(1, 1, 1, encoder_outputs.size(3))

        # make context vector by element wise mul
        context = alignment * encoder_outputs
        context = torch.mean(context, 2)

        return context

    @staticmethod
    def combine_pos_and_word_embedding(words, positions):
        """
        Define the way positional and word embeddings are combined on the "encoder" side.
        """
        return torch.cat((words, positions), 2)

    @staticmethod
    def load(model_path):
        return torch.load(model_path)

    def save(self, model_path):
        torch.save(self, model_path)
