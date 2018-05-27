"""
Module defining all the evaluation functions for the model.
"""

# STD
import codecs
import subprocess

# EXT
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F

# PROJECT
from corpus import ParallelCorpus
from seq2seq import Seq2Seq, Encoder


def evaluate(encoder, decoder, eval_set, target_path, reference_path="./reference.en"):
    data_loader = DataLoader(eval_set, batch_size=5)
    idx2word = evaluation_set.target_i2w
    sorted_sentence_ids = evaluation_set.target_sentence_ids.cpu().numpy()
    translated_sentences = []

    # Decode
    # for source_batch, target_batch, source_lengths, target_lengths, batch_positions in data_loader:
    for source_batch, target_batch, source_lengths, target_lengths, batch_positions in data_loader:
        encoder_out, h_t = encoder(
            input_src=source_batch, src_lengths=source_lengths, positions=batch_positions,
        )
        decoder_out = decoder(
            encoder_out=encoder_out, h_t=h_t,
            input_trg=target_batch, source_lengths=source_lengths, teacher=True
        )

        # Get predicted word for every batch instance
        #normalized_output = F.softmax(decoder_out, dim=2)
        predictions = decoder_out.max(2)[1]  # Only get indices

        for sentence_index in range(predictions.shape[0]):
            token_indices = list(predictions[sentence_index, :].numpy())

            tokens = list(map(lambda idx: idx2word[idx], token_indices))
            eos_index = len(tokens)
            if "<eos>" in tokens:
                eos_index = tokens.index("<eos>")

            tokens = tokens[:eos_index]  # Cut off after first end of sentence token

            translated_sentence = " ".join(tokens).replace("@@ ", "").replace("@@", "")
            print(translated_sentence)
            translated_sentences.append(translated_sentence)

    # Bring sentence back into the order they were in the test set
    resorted_sentences = [None] * len(translated_sentences)
    for target_id, sentence in zip(sorted_sentence_ids, translated_sentences):
        resorted_sentences[target_id] = sentence

    # Write to file
    with codecs.open(target_path, "wb", "utf-8") as target_file:
        for sentence in resorted_sentences:
            target_file.write("{}\n".format(sentence))

    # Write reference file
    with codecs.open(reference_path, "wb", "utf-8") as reference_file:
        for sentence in evaluation_set.target_sentences:
            sentence = " ".join(sentence).replace("@@ ", "").replace("@@", "")
            reference_file.write("{}\n".format(sentence))

    out = subprocess.getoutput(
        "perl ./multi-bleu.perl {} < {}".format(reference_path, target_path)
    )
    print(out[out.index("BLEU"):])


if __name__ == "__main__":
    max_allowed_sentence_len = 50
    training_set = ParallelCorpus(
        source_path="./data/train/train_bpe.fr", target_path="./data/train/train_bpe.en",
        max_sentence_length=max_allowed_sentence_len
    )
    evaluation_set = ParallelCorpus(
        source_path="./data/test/test_2017_flickr_bpe.fr", target_path="./data/test/test_2017_flickr_bpe.en",
        max_sentence_length=max_allowed_sentence_len, use_indices_from=training_set
    )

    encoder = torch.load("./encoder_cpu.model")
    decoder = torch.load("./decoder_cpu.model")

    evaluate(
        encoder, decoder, evaluation_set, target_path="./eval_out.txt"
    )


