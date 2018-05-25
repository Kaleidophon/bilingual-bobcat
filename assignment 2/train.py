"""
Module defining the model training function.
"""

# STD
import time
import random

# EXT
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
#np.set_printoptions(threshold=np.nan)

# PROJECT
from corpus import ParallelCorpus
from models import Encoder, Decoder


def train(encoder, decoder, num_epochs, loss_function, optimizer, target_dim, force, iterations, save_dir=None,
          dataset=None, debug=False, clip=0.25):
    epoch_losses = []

    for epoch in range(0, num_epochs):
        start = time.time()
        batch_loss = 0
        batch = 0

        for source_batch, target_batch, source_lengths, target_lengths, batch_positions in data_loader:
            batch_start = time.time()
            batch_size = source_batch.size(0)

            if debug:
                source_batch_sentences = source_batch.numpy()
                source_batch_sentences = np.array([
                    np.array(list(map(lambda tid: dataset.source_i2w[tid], source_batch_sentences[i])))
                    for i in range(source_batch_sentences.shape[0])
                ])
                target_batch_sentences = target_batch.numpy()
                target_batch_sentences = np.array([
                    np.array(list(map(lambda tid: dataset.target_i2w[tid], target_batch_sentences[i])))
                    for i in range(target_batch_sentences.shape[0])
                ])
                print(
                    "Source sentences\n", source_batch_sentences,
                    "\nTarget sentences:\n", target_batch_sentences,
                    "\nSource lengths:\n", source_lengths,
                    "\nTarget lengths:\n", target_lengths,
                    "\nPositions:\n", batch_positions,
                    "\n"
                )

            max_len = source_lengths.max()
            target_len = target_lengths.max()
            # target_batch = target_batch[:, :target_len]

            source_batch = torch.autograd.Variable(source_batch)
            target_batch = torch.autograd.Variable(target_batch)
            source_lengths = torch.autograd.Variable(source_lengths)
            target_lengths = torch.autograd.Variable(target_lengths)
            batch_positions = torch.autograd.Variable(batch_positions)

            if torch.cuda.is_available():
                source_batch = source_batch.cuda()
                target_batch = target_batch.cuda()
                source_lengths = source_lengths.cuda()
                target_lengths = target_lengths.cuda()
                batch_positions = batch_positions.cuda()

            loss = 0

            # get encoder outputs
            #encoder_out, previous_hidden = encoder(source_batch, source_lengths)
            encoder_out, previous_hidden, padded_positions = encoder(source_batch, batch_positions)

            use_teacher_forcing = True if random.random() <= force else False

            if use_teacher_forcing:
                bloss = 0
                for i in range(target_len-1):
                    word_batch = target_batch[:, i]  # Current correct tokens
                    #decoder_out, current_hidden = decoder(
                    #    word_batch, previous_hidden, encoder_out
                    #)
                    decoder_out, current_hidden = decoder(
                        word_batch, encoder_out, max_len, previous_hidden=previous_hidden,
                        padded_positions=padded_positions
                    )
                    bloss += loss_function(decoder_out, target_batch[:, i + 1])
                    previous_hidden = current_hidden

                    # TODO: Remove this debugging printing
                    #print("Out", F.softmax(decoder_out))
                    if debug:
                        _, words = decoder_out.topk(1)
                        decoder_in = words.squeeze().detach()
                        decoder_word_predictions = np.array([dataset.target_i2w[idx] for idx in decoder_in.numpy()])
                        overlap = target_batch_sentences[:, i].T == decoder_word_predictions
                        print(
                            "Given \n", target_batch_sentences[:, i].T,
                            "\nTry to predict\n", target_batch_sentences[:, i + 1].T,
                            "\nPredicted:\n", decoder_word_predictions,
                            "\nCorrectly predicted: {:.2f} %".format((list(overlap).count(True) / len(overlap) * 100)),
                            "\n\n\n"
                        )
                #bloss /= batch_size
                loss += bloss
            else:
                decoder_in = target_batch[:, 0]
                for i in range(target_len-1):
                    # TODO: METHOD TO LIMIT TO SENTENCES STILL ACTIVE IN i (ie NOT PADDING RIGHT NOW)
                    # PERHAPS NOT NEEDED AS LOSS INGNORES PADS ANYWAY???
                    decoder_out, current_hidden = decoder(
                        decoder_in, previous_hidden, encoder_out
                    )
                    loss += loss_function(decoder_out, target_batch[:, i+1])
                    _, words = decoder_out.topk(1)
                    decoder_in = words.squeeze().detach()  # this is not tested.
                    previous_hidden = current_hidden

            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

            optimizer.step()
            optimizer.zero_grad()

            batch_loss += loss.item()

            batch_time = time.time() - batch_start

            if debug:
                # TODO: Remove
                print("\n\n\n")
            print('\r[Epoch {:03d}/{:03d}] Batch {:06d}/{:06d} [{:.1f}/s] '.format(epoch + 1, num_epochs, batch + 1,
                                                                                   iterations, batch_time), end='')
            batch += 1

        print('Time: {:.1f}s Loss: {:.3f}'.format(time.time() - start, batch_loss))

        if save_dir is not None:
            torch.save(encoder, "{}{}_epoch{}.model".format(save_dir, encoder.__class__.__name__.lower(), epoch+1))
            torch.save(decoder, "{}{}_epoch{}.model".format(save_dir, decoder.__class__.__name__.lower(), epoch+1))


if __name__ == "__main__":
    # Define hyperparameters
    num_epochs = 10
    batch_size = 128
    learning_rate = 0.01
    embedding_dim = 100
    hidden_dim = 2 * embedding_dim
    max_allowed_sentence_len = 50
    force = 1

    # Prepare training
    training_set = ParallelCorpus(
        source_path="./data/train/train_bpe.fr", target_path="./data/train/train_bpe.en",
        max_sentence_length=max_allowed_sentence_len
    )
    data_loader = DataLoader(training_set, batch_size=batch_size)
    loss_function = nn.CrossEntropyLoss(ignore_index=training_set.target_pad)
    iterations = len(data_loader)

    encoder = Encoder(
        source_vocab_size=training_set.source_vocab_size, embedding_dims=embedding_dim,
        max_sentence_len=max_allowed_sentence_len, hidden_dims=hidden_dim, pad_index=training_set.source_pad
    )

    decoder = Decoder(
        target_vocab_size=training_set.target_vocab_size,
        embedding_dims=embedding_dim, hidden_dims=hidden_dim, max_length=max_allowed_sentence_len
    )
    if torch.cuda.is_available():
        loss_function = loss_function.cuda()
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    # Train
    train(
        encoder, decoder, num_epochs, loss_function, optimizer, training_set.target_vocab_size, force, iterations,
        save_dir="./", dataset=training_set, debug=True
    )

