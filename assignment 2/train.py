"""
Module defining the model training function.
"""

# STD
import time

# EXT
import torch
from torch import nn
from torch.utils.data import DataLoader

# PROJECT
from corpus import ParallelCorpus
from models import AttentionModel


def train(model, num_epochs, loss_function, optimizer, save_dir=None):
    epoch_losses = []
    iterations = len(data_loader)

    for epoch in range(0, num_epochs):
        start = time.time()
        batch_losses = []
        batch = 0

        for source_batch, target_batch, source_lengths, target_lengths, batch_positions in data_loader:
            batch_start = time.time()
            max_len = int(source_lengths.max().numpy())
            source_batch = source_batch[:, :max_len]
            batch_positions = batch_positions[:, :max_len]
            target_len = target_lengths.max()

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

            combined_embeddings, hidden = model.encoder_forward(source_batch, source_lengths, batch_positions)

            for target_pos in range(target_len):
                current_target_words = target_batch[:, target_pos]
                model.decoder_forward(current_target_words, hidden, combined_embeddings, source_lengths, max_len)


                # # get encoder and decoder outputs
                # decoder_out = model.forward(
                #     source_batch, source_lengths, batch_positions, target_batch, target_lengths
                # )
                #
                # # remove the start token from the targets and the end token from the decoder
                # decoder_out2 = decoder_out[:, :-1, :]
                # target_batch2 = target_batch[:, 1:decoder_out.size(1)]
                #
                # # pytorch expects the inputs for the loss function to be: (batch x classes)
                # # cant handle sentences so we just transform our data to treath each individual word as a batch:
                # # so new batch size = old batch * padded sentence length
                # # should not matter as it all gets averaged out anyway
                # decoder_out3 = decoder_out2.contiguous().view(decoder_out2.size(0) * decoder_out2.size(1), decoder_out2.size(2))
                # target_batch3 = target_batch2.contiguous().view(target_batch2.size(0) * target_batch2.size(1))
                #
                # # calculate loss
                # loss = loss_function(decoder_out3, target_batch3)
                # batch_losses.append(loss)
                #
                # # backward and optimize
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                #
                # batch_time = time.time() - batch_start
                # print('\r[Epoch {:03d}/{:03d}] Batch {:06d}/{:06d} [{:.1f}/s] '.format(epoch + 1, num_epochs, batch + 1,
                #                                                                        iterations, batch_time), end='')
                # batch += 1

        avg_loss = sum(batch_losses) / iterations
        epoch_losses.append(avg_loss)
        print('Time: {:.1f}s Loss: {:.3f} score2?: {:.6f}'.format(time.time() - start, avg_loss, 0))

        if save_dir is not None:
            model.save("{}{}_epoch{}.model".format(save_dir, model.__class__.__name__.lower(), epoch+1))


if __name__ == "__main__":
    # Define hyperparameters
    num_epochs = 30
    batch_size = 128
    learning_rate = 0.01
    embedding_dim = 100
    hidden_dim = embedding_dim * 2
    max_allowed_sentence_len = 50

    # Prepare training
    training_set = ParallelCorpus(
        source_path="./data/train/train_bpe.fr", target_path="./data/train/train_bpe.en",
        max_sentence_length=max_allowed_sentence_len
    )
    data_loader = DataLoader(training_set, batch_size=batch_size)
    loss_function = nn.CrossEntropyLoss(ignore_index=training_set.target_pad).cpu()
    max_corpus_sentence_len = training_set.source_lengths.max().numpy()

    # Init model
    model = AttentionModel(
        source_vocab_size=training_set.source_vocab_size, target_vocab_size=training_set.target_vocab_size,
        embedding_dim=embedding_dim, hidden_dim=hidden_dim, encoded_positions=51
    )
    optimizer = torch.optim.Adam(list(model.parameters()), learning_rate)

    # Train
    train(model, num_epochs, loss_function, optimizer, save_dir="./")

