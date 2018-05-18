"""
Main module defining the translation model.
"""

# EXT
import numpy as np
import torch.nn as nn


class AttentionModel(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, encoded_positions=100,
                 attention_func=np.dot, embedding_comb_func=np.concatenate):
        """
        Create a new translation model with an attention mechanism and positional embeddings
        on the encoder side and a decoder.

        :param vocab_size: Size of the vocabulary of the source language
        :param embedding_dim: Dimensionality of word and positional embeddings
        :param hidden_dim: Dimensionality of hidden states on the decoder side.
        :param encoded_positions: Number of positions in the input sentence that positional
        embeddings will be trained for.
        :param attention_func: Kind of function used a similarity measure for the attention mechanism.
        :param embedding_comb_func: Kind of function used to combine word and positional_embeddings
        """
        super().__init__()
        self.num_encoded_positions = encoded_positions
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.attention_func = attention_func
        self.embedding_comb_func = embedding_comb_func

        # Define weights
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embeddings = nn.Embedding(encoded_positions, embedding_dim)
        # Weight matrix hidden state to hidden state on decoder side
        self.hidden2hidden = nn.Linear(hidden_dim, hidden_dim)
        # Weight matrix hidden state to output on decoder side
        self.hidden2output = nn.Linear(hidden_dim, embedding_dim)
        # Weight matrix previous output and context vector from attention mechanism to
        # hidden state on decoder side
        self.output2hidden = nn.Linear(2 * embedding_dim, hidden_dim)

    def forward(self, *input):
        """
        Forward pass through the model.
        """
        # TODO: Implement
        ...

    def attention(self, *input):
        # TODO: Implement
        ...

    def combine_pos_and_word_embedding(self, word_embedding, pos_embedding):
        return self.embedding_comb_func(word_embedding, pos_embedding)

    def load(self, model_path):
        # TODO: Implement
        ...

    def save(self, model_path):
        # TODO: Implement
        ...
