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
    def __init__(self, target_vocab_size, embedding_dims, hidden_dims):
        super(Decoder, self).__init__()

        # Decoder stuff
        self.decoder_embeddings = nn.Embedding(target_vocab_size, embedding_dims)
        self.lstm = nn.LSTM(embedding_dims*3, hidden_dims, batch_first=True)
        self.scale_h0 = nn.Linear(embedding_dims * 2, hidden_dims)
        self.out = nn.Linear(embedding_dims * 2, target_vocab_size)

        # Attention stuff

        # used only in the concat version
        self.attn_layer = nn.Linear(embedding_dims * 2 + hidden_dims, 1)

        self.attn_soft = nn.Softmax(dim=2)

    def forward(self, hidden=None, target_sentences, target_lengths, source_lengths, encoder_outputs, max_len):
        words = self.decoder_embeddings(target_sentences)

        lstm_out = words[:,0,:].unsqueeze(1)

        out_list = []
        for i in range(target_len):
            #print(" ")
            # choose right input based on force or not
            if (i==0 or force):
                lstm_in = words[:,i,:].unsqueeze(1)
            else:
                #print("AAAAAAAAA")
                # out should be done in loop....
                #print(lstm_out.size())
                _, word = torch.max(lstm_out,2)
                #print(word.size())
                lstm_in = self.decoder_embeddings(word)
                #print(lstm_in.size())


            # get the context vectors
            # print("hidden input ", hidden[0].size())

            
            #context = self.attention(hidden[0].squeeze().unsqueeze(1), encoder_outputs, source_lengths, max_len)
            # print("context out ", context.size())
            # print(lstm_in.size())
            lstm_in = torch.cat((lstm_in, lstm_in,lstm_in),2)
            # print(hidden[0].size())
            lstm_out, hidden = self.lstm(lstm_in, hidden)
            # print(hidden[0].size())
            lstm_out = self.out(lstm_out)
            out_list.append(lstm_out.squeeze())

        
        out = torch.stack(out_list,1)

        return out, hidden

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

        idxes = torch.arange(0, max_len).unsqueeze(0).long()

        if torch.cuda.is_available():
            idxes = idxes.cuda()

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

class Encoder(nn.Module):
    def __init__(self, source_vocab_size, max_sentence_len, embedding_dims, hidden_dims):
        super(Encoder, self).__init__())
        self.encoder_embeddings = nn.Embedding(source_vocab_size, embedding_dims)  # embed words
        self.pos_embeddings = nn.Embedding(max_sentence_len + 1, embedding_dims)  # embed positions
        self.scale_h0 = nn.Linear(embedding_dims * 2, hidden_dims)
    def forward(self, source_sentences, positions):
        words = self.encoder_embeddings(source_sentences)
        pos = self.pos_embeddings(positions)
        cat = torch.cat((words, pos), 2)


        avg = torch.mean(cat, 1)
        hidden = self.scale_h0(avg)
        hidden = torch.unsqueeze(hidden, 0)
        hidden = (hidden,hidden)

        return cat, hidden
