"""
Main module defining the translation model.
"""

# EXT
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Decoder(nn.Module):
    def __init__(self, source_vocab_size, max_sentence_len, target_vocab_size, embedding_dims, hidden_dims):
        super(Decoder, self).__init__()

        # Encoder stuff
        self.encoder_embeddings = nn.Embedding(source_vocab_size, embedding_dims)  # embed words
        self.pos_embeddings = nn.Embedding(max_sentence_len + 1, embedding_dims)  # embed positions

        # Decoder stuff
        self.decoder_embeddings = nn.Embedding(target_vocab_size, embedding_dims)
        self.lstm = nn.LSTM(embedding_dims, hidden_dims, batch_first=True)
        self.scale_h0 = nn.Linear(embedding_dims * 2, hidden_dims)
        self.out = nn.Linear(embedding_dims * 2 + hidden_dims, target_vocab_size)

        # Attention stuff

        # used only in the concat version
        self.attn_layer = nn.Linear(embedding_dims * 2 + hidden_dims, 1)

        self.attn_soft = nn.Softmax(dim=2)

    def forward(self, target_sentences, target_lengths, source_lengths, source_sentences, positions, max_len):
        words = self.decoder_embeddings(target_sentences)

        encoder_outputs = self.encoder(source_sentences, positions)

        # avg out encoder data to use as hidden state
        avg = torch.mean(encoder_outputs, 1)
        hidden = self.scale_h0(avg)

        # get hidden state from encoder RNN
        packed_input = pack_padded_sequence(words, target_lengths, batch_first=True)
        packed_output, (ht, ct) = self.lstm(packed_input, (torch.unsqueeze(hidden, 0), torch.unsqueeze(hidden, 0)))
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)

        # prepare lstm Hidden layers for input into attention,
        # we add the hidden layer as zeroth lstm_out and remove the last one
        # this is because we need h_i-1, not h_i
        lstm_to_attn = torch.cat((torch.unsqueeze(hidden, 1), lstm_out), 1)
        lstm_to_attn = lstm_to_attn = lstm_to_attn[:, :-1, :]

        context = self.attention(lstm_to_attn, encoder_outputs, source_lengths, max_len)

        # combine with non existing context vectors
        combined = torch.cat((lstm_out, context), 2)
        out = self.out(combined)
        return out

    def attention(self, lstm_to_attn, encoder_outputs, source_lengths, max_len):
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
        allignment = self.attn_soft(energy)

        # create a mask like : [1,1,1,0,0,0] whos goal is to multiply the attentions of the pads with 0, rest with 1
        idxes = torch.arange(0, max_len, out=max_len).unsqueeze(0)
        mask = Variable((idxes < source_lengths.unsqueeze(1)).float())

        # format the mask to be same size() as the attentions
        mask = mask.unsqueeze(1).unsqueeze(3).repeat(1, allignment.size(1), 1, 1)

        # apply mask
        masked = allignment * mask

        # now we have to rebalance the other values so they sum to 1 again
        # this is done by dividing each value by the sum of the sequence
        # calculate sums
        msum = masked.sum(-2).repeat(1, 1, masked.size(2)).unsqueeze(3)

        # rebalance
        attentions = masked.div(msum)

        # now we shape the attentions to be similar to context in size
        allignment = allignment.repeat(1, 1, 1, encoder_outputs.size(3))

        # make context vector by element wise mul
        context = attentions * encoder_outputs

        context2 = torch.sum(context, 2)

        return context2

    def encoder(self, source_sentences, positions):
        words = self.encoder_embeddings(source_sentences)
        pos = self.pos_embeddings(positions)
        return torch.cat((words, pos), 2)
