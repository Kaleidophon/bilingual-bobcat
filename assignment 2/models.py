"""
Main module defining the translation model.
"""

# EXT
import numpy as np
import torch.nn as nn


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
        self.word_embeddings = nn.Embedding(self.source_vocab_size, embedding_dim)
        self.positional_embeddings = nn.Embedding(encoded_positions, embedding_dim)
        self.attention_layer = nn.Linear(embedding_dim * 2, 1)
        # Projecting the context vector (which is the weighted average over concats of positional and word embeddings)
        # concatenated with the current hidden unit to the target vocabulary in order to apply softmax
        self.projection_layer = nn.Linear(2 * embedding_dim + hidden_dim, target_vocab_size)
        # Hidden units on decoder side
        self.hidden_layer = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, dropout=self.dropout)

    def forward(self, *input):
        """
        Forward pass through the model.
        """
        # TODO: Implement
        ...

    def attention(self, *input):
        """
        Attention mechanism.
        """
        # TODO: Implement dot product first, then with self.attention_layer
        ...

    def combine_pos_and_word_embedding(self, word_embedding, pos_embedding):
        """
        Define the way positional and word embeddings are combined on the "encoder" side.
        """
        return self.embedding_comb_func(word_embedding, pos_embedding)

    def load(self, model_path):
        # TODO: Implement
        ...

    def save(self, model_path):
        # TODO: Implement
        ...
