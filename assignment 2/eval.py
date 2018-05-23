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
    # for source_batch, target_batch, source_lengths, target_lengths, batch_positions in data_loader:
    for source_batch, target_batch, source_lengths, target_lengths, batch_positions in data_loader:
        # TODO: Don't use target sentences for prediction!
        max_len = source_lengths.max()
        output = model(target_batch, target_lengths, source_lengths, source_batch, batch_positions, max_len)
        normalized_output = softmax.forward(output)
        predictions = normalized_output.max(2)[1].cpu().numpy()  # Only get indices
        _, sentence = torch.max(normalized_output[0], 1)
        #print([idx2word[word] for word in sentence.cpu().numpy()])
        #print([idx2word[word] for word in target_batch[0].cpu().numpy()])
        # print([eval_set.source_i2w[word] for word in source_batch[0].cpu().numpy()])
        #print(" ")

        for sentence_index in range(predictions.shape[0]):
            token_indices = predictions[sentence_index]
            tokens = list(map(lambda idx: idx2word[idx], token_indices))

            eos_index = len(tokens)
            if "<eos>" in tokens:
                eos_index = tokens.index("<eos>")

            tokens = tokens[:eos_index]  # Cut off after first end of sentence token
            # TODO: This is only for debugging: Add sentence id and real sentence ids
            prefix = ["Sentence id: {}".format(sentence_index), "Real id: {}".format(sorted_sentence_ids[sentence_index])]
            prefix.extend(tokens)
            tokens = prefix

            translated_sentence = " ".join(tokens).replace("@@ ", "")
            translated_sentences.append(translated_sentence)
            # print(translated_sentence)
            # print([idx2word[word] for word in target_batch[sentence_index].cpu().numpy()])
            # print(" ")

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


if __name__ == "__main__":
    model = torch.load("./decoder30.model")
    max_allowed_sentence_len = 50
    training_set = ParallelCorpus(
        source_path="./data/train/train.fr", target_path="./data/train/train.en",
        max_sentence_length=max_allowed_sentence_len
    )
    evaluation_set = ParallelCorpus(
        source_path="./data/test/test_2017_flickr.fr", target_path="./data/test/test_2017_flickr.en",
        max_sentence_length=max_allowed_sentence_len, use_indices_from=training_set
    )
    evaluate(
        model, evaluation_set, target_path="./eval_out.txt",
        reference_file_path="./data/test/test_2017_flickr_truecased.en"
    )

