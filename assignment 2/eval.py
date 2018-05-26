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

# PROJECT
from corpus import ParallelCorpus
from seq2seq import Seq2Seq


def evaluate(model, eval_set, target_path, reference_file_path):
    data_loader = DataLoader(eval_set, batch_size=5)
    softmax = nn.Softmax(dim=1)
    idx2word = evaluation_set.target_i2w
    sorted_sentence_ids = evaluation_set.target_sentence_ids.cpu().numpy()
    translated_sentences = []

    # Decode
    # for source_batch, target_batch, source_lengths, target_lengths, batch_positions in data_loader:
    for source_batch, target_batch, source_lengths, target_lengths, batch_positions in data_loader:
        decoder_out = model(
            input_src=source_batch, input_trg=target_batch, src_lengths=source_lengths
        )

        # Get predicted word for every batch instance
        normalized_output = softmax(decoder_out)
        predictions = normalized_output.max(2)[1]  # Only get indices

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

    out = subprocess.getoutput(
        "perl ./multi-bleu.perl {} < {}".format(reference_file_path, target_path)
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

    model = torch.load("./seq2seq_10_alpha_cpu.model")

    evaluate(
        model, evaluation_set, target_path="./eval_out.txt",
        reference_file_path="./data/test/test_2017_flickr_truecased.en"
    )


