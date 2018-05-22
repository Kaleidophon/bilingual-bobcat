"""
Module defining all the evaluation functions for the model.
"""

# STD
import codecs
import subprocess
import sys
import os
from io import StringIO
import time

# EXT
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

# PROJECT
from models import AttentionModel
from corpus import ParallelCorpus


def evaluate(model, eval_set, target_path, reference_file_path):
    data_loader = DataLoader(eval_set, batch_size=5)
    softmax = nn.Softmax(dim=2)
    idx2word = evaluation_set.target_i2w
    sorted_sentence_ids = evaluation_set.target_sentence_ids.cpu().numpy()
    translated_sentences = []

    # Decode
    for source_batch, _1, source_lengths, _2, batch_positions in data_loader:
        # TODO: Using the model's prediction only produces <bos> tokens
        # TODO: Clean up
        combined_embeddings, hidden = model.encoder_forward(source_batch, batch_positions)
        target_len = _2.max()
        max_len = source_lengths.max().cpu().numpy()
        batch_size = source_batch.size(0)
        # Initialize first hidden layer input
        # 2 = <bos> token
        target_words = torch.LongTensor(np.array([2] * batch_size))
        if torch.cuda.is_available():
            target_words = target_words.cuda()
        predicted_word_indices = []

        for target_pos in range(target_len):
            print("tar input", target_words.size(), _1[:, target_pos].size())
            # Do the forward pass
            out, hidden = model.decoder_forward(
                target_words, hidden, combined_embeddings, source_lengths, max_len
            )
            print(out.size())
            out = out.unsqueeze(1)
            print("out", out)

            # Get predicted words
            normalized_output = softmax.forward(out)
            print("Norm out", normalized_output.size(), normalized_output)
            _, predictions = normalized_output.max(2)
            print("Predictions", predictions)

            # Use predicted words as next input
            target_words = predictions.squeeze(1)
            print("tar", target_words.size())

            # Save the predicted indices for later
            predicted_word_indices.append(predictions)

        predicted_word_indices = torch.cat(predicted_word_indices, 1)

        for sentence_index in range(predicted_word_indices.shape[0]):
            token_indices = predicted_word_indices[sentence_index]
            print(token_indices)
            tokens = list(map(lambda idx: idx2word[idx], token_indices))

            eos_index = len(tokens)
            if "<eos>" in tokens:
                eos_index = tokens.index("<eos>")

            tokens = tokens[:eos_index]  # Cut off after first end of sentence token
            translated_sentence = " ".join(tokens).replace("@@ ", "")
            print(translated_sentence)
            translated_sentences.append(translated_sentence)

    # Bring sentence back into the order they were in the test set
    translated_sentences = np.array(translated_sentences)[sorted_sentence_ids]

    # Write to file
    with codecs.open(target_path, "wb", "utf-8") as target_file:
        for sentence in translated_sentences:
            target_file.write("{}\n".format(sentence))

    out = subprocess.getoutput(
        "perl ./multi-bleu.perl {} < {}".format(reference_file_path, target_path)
    )
    print(out[out.index("BLEU"):])

    # TODO
    # In terminal: Download and apply METEOR
    # http://www.cs.cmu.edu/~alavie/METEOR/


if __name__ == "__main__":
    model = AttentionModel.load("./attentionmodel_epoch2.model")
    max_allowed_sentence_len = 50
    training_set = ParallelCorpus(
        source_path="./data/train/train_bpe.fr", target_path="./data/train/train_bpe.en",
        max_sentence_length=max_allowed_sentence_len
    )
    evaluation_set = ParallelCorpus(
        source_path="./data/test/test_2017_flickr_bpe.fr", target_path="./data/test/test_2017_flickr_bpe.en",
        max_sentence_length=max_allowed_sentence_len, use_indices_from=training_set
    )
    evaluate(
        model, evaluation_set, target_path="./eval_out.txt",
        reference_file_path="./data/test/test_2017_flickr_truecased.en"
    )
