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
        self.attention_soft = nn.Softmax(dim=1)
        # Projecting the context vector (which is the weighted average over concats of positional and word embeddings)
        # concatenated with the current hidden unit to the target vocabulary in order to apply softmax
        self.projection_layer = nn.Linear(2 * embedding_dim + hidden_dim, target_vocab_size)
        # Hidden units on decoder side
        self.lstm = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.scale_h0 = nn.Linear(embedding_dim * 2, hidden_dim)

        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

    def encoder_forward(self, source_sentences, source_lengths, positions, target_sentences=None, target_lengths=None):
        """
        Forward pass through the model.
        """
        max_len = source_lengths.max()

        # Encoder side
        in_words = self.word_embeddings_in(source_sentences)
        positions = self.positional_embeddings(positions)
        combined_embeddings = self.combine_pos_and_word_embedding(in_words, positions)

        # avg out encoder data to use as hidden state
        avg = torch.mean(combined_embeddings, 1)
        hidden0 = self.scale_h0(avg)

        return combined_embeddings, hidden0

    def decoder_forward(self, target_words, last_hidden, encoder_output, source_lengths, max_len):
        # In contrast to the encoder, don't get the embeddings for all words of all the batch sentences here,
        # but instead just the embeddings for all the words at a certain position in the current batch
        out_words = self.word_embeddings_out(target_words)

        context = self.attention(last_hidden, encoder_output, source_lengths, max_len)

        return None
        #return out, hidden

    def attention(self, current_hidden, encoder_outputs, source_lengths, max_len):
        """
        Defining the Attention mechanism.
        """
        batch_size, dim = current_hidden.size()
        energy = Variable(torch.zeros(batch_size, max_len))

        for i in range(max_len):
            batch_size, dim = current_hidden.size()
            #energy[i] = encoder_outputs.unsqueeze(2).matmul(current_hidden.unsqueeze(3)).squeeze(3)
            ch = current_hidden.view(batch_size, 1, dim)
            eo = encoder_outputs[:, i, :].view(batch_size, dim, 1)
            energy[:, i] = torch.bmm(ch, eo).squeeze(2).squeeze(1)

        # apply softmax to the energys
        alignment = self.attention_soft(energy).unsqueeze(2)

        print("Encoder output", encoder_outputs.size())
        print("align", alignment.size())

        # make context vector by element wise mul
        context = alignment * encoder_outputs
        print("SUmm context", context.size())
        context = torch.mean(context, 1)
        print("mean context", context.size())

        return context

    @staticmethod
    def get_onehots(index_tensor, vocabulary_size):
        print(index_tensor)
        print(index_tensor.size())
        inp = index_tensor % vocabulary_size
        inp_ = torch.unsqueeze(inp, 2)

        one_hots = torch.FloatTensor(index_tensor.size(0), index_tensor.size(1), vocabulary_size).zero_()
        one_hots.scatter_(2, inp_, 1)
        return one_hots

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
