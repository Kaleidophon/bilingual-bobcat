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

# PROJECT
from models import AttentionModel
from corpus import ParallelCorpus


def evaluate(model, eval_set, target_path, reference_file_path):
    data_loader = DataLoader(eval_set, batch_size=5)
    softmax = nn.Softmax(dim=2)
    idx2word = evaluation_set.target_i2w
    sorted_sentence_ids = evaluation_set.target_sentence_ids.numpy()
    translated_sentences = []

    # Decode
    for source_batch, _1, source_lengths, _2, batch_positions in data_loader:
        # TODO: Don't use target sentences for prediction!
        output = model.forward(source_batch, source_lengths, batch_positions, _1, _2)
        normalized_output = softmax.forward(output)
        predictions = normalized_output.max(2)[1].numpy()  # Only get indices

        for sentence_index in range(predictions.shape[0]):
            token_indices = predictions[sentence_index]
            tokens = list(map(lambda idx: idx2word[idx], token_indices))

            eos_index = len(tokens)
            if "<eos>" in tokens:
                eos_index = tokens.index("<eos>")

            tokens = tokens[:eos_index]  # Cut off after first end of sentence token
            translated_sentence = " ".join(tokens).replace("@@ ", "")
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
    model = AttentionModel.load("./attentionmodel_epoch30.model")
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
