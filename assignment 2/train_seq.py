from __future__ import print_function
import time
import numpy as np
#import numpy as np
#import codecs
#import nltk
import torch
from corpus import ParallelCorpus
from torch.utils.data import DataLoader
#from seq2seq import Seq2Seq


if __name__ == "__main__":
    num_epochs = 10
    batch_size = 64
    learning_rate = 4e-4
    embedding_dim = 512
    #hidden_dim = 2 * embedding_dim
    max_allowed_sentence_len = 50
    #force = 1
    volatile = False

    cuda_available = torch.cuda.is_available()

    training_set = ParallelCorpus(
        source_path="./data/train/train_bpe.fr", target_path="./data/train/train_bpe.en",
        max_sentence_length=max_allowed_sentence_len
    )

    #valuation_set = ParallelCorpus(
    #    source_path="./data/val/val_bpe.fr", target_path="./data/val/val_bpe.en",
    #    max_sentence_length=max_allowed_sentence_len, use_indices_from=training_set
    #)

    test_set = ParallelCorpus(
        source_path="./data/test/test_2017_flickr_bpe.fr", target_path="./data/test/test_2017_flickr_bpe.en",
        max_sentence_length=max_allowed_sentence_len, use_indices_from=training_set
    )
