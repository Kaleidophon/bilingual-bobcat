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
from models import Decoder


def train(model, num_epochs, loss_function, optimizer, target_dim, force, iterations, save_dir=None):
    epoch_losses = []

    for epoch in range(0, num_epochs):
        start = time.time()
        batch_losses = []
        batch = 0

        for source_batch, target_batch, source_lengths, target_lengths, batch_positions in data_loader:
            batch_start = time.time()
            max_len = source_lengths.max()
            source_batch = source_batch[:, :max_len]
            batch_positions = batch_positions[:, :max_len]
            target_len = target_lengths.max()
            target_batch = target_batch[:,:target_len]

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

            # get encoder and decoder outputs
            # encoder_out = encoder(source_batch, batch_positions)
            decoder_out = model.forward(target_batch, target_lengths, source_lengths, source_batch, batch_positions, max_len, target_len, force)

            # remove the start token from the targets and the end token from the decoder
            decoder_out2 = decoder_out[:, :-1, :]
            target_batch2 = target_batch[:, 1:decoder_out.size(1)]

            # pytorch expects the inputs for the loss function to be: (batch x classes)
            # cant handle sentences so we just transform our data to treath each individual word as a batch:
            # so new batch size = old batch * padded sentence length
            # should not matter as it all gets averaged out anyway
            decoder_out3 = decoder_out2.contiguous().view(decoder_out2.size(0) * decoder_out2.size(1),
                                                          decoder_out2.size(2))
            target_batch3 = target_batch2.contiguous().view(target_batch2.size(0) * target_batch2.size(1))

            # calculate loss
            # print(decoder_out2.size())
            # print(target_batch[0])
            #         _, sentence = torch.max(decoder_out2[0],1)
            #         test_pred = [dataset.target_i2w[word] for word in sentence.cpu().numpy()]
            #         print(test_pred)
            #         test_real = [dataset.target_i2w[word] for word in target_batch2[0].cpu().numpy()]
            #         print(test_real)

            loss = loss_function(decoder_out3, target_batch3)
            #print(loss)
            batch_losses.append(loss)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss)
            #print(batch_losses)
            batch_time = time.time() - batch_start
            print('\r[Epoch {:03d}/{:03d}] Batch {:06d}/{:06d} [{:.1f}/s] '.format(epoch + 1, num_epochs, batch + 1,
                                                                                   iterations, batch_time), end='')
            batch += 1
        #print(len(batch_losses)), iterations
        avg_loss = sum(batch_losses) / iterations
        #print(avg_loss)
        epoch_losses.append(avg_loss)
        print('Time: {:.1f}s Loss: {:.3f} score2?: {:.6f}'.format(time.time() - start, avg_loss, 0))

        if save_dir is not None:
            torch.save(model, "{}{}_epoch{}.model".format(save_dir, model.__class__.__name__.lower(), epoch+1))


if __name__ == "__main__":
    # Define hyperparameters
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.01
    embedding_dim = 100
    hidden_dim = 2 * embedding_dim
    max_allowed_sentence_len = 50
    force = True

    # Prepare training
    training_set = ParallelCorpus(
        source_path="./data/train/train_bpe.fr", target_path="./data/train/train_bpe.en",
        max_sentence_length=max_allowed_sentence_len
    )
    data_loader = DataLoader(training_set, batch_size=batch_size)
    loss_function = nn.CrossEntropyLoss(ignore_index=training_set.target_pad)
    iterations = len(data_loader)
    # Init model
    model = Decoder(
        source_vocab_size=training_set.source_vocab_size, target_vocab_size=training_set.target_vocab_size,
        embedding_dims=embedding_dim, hidden_dims=hidden_dim, max_sentence_len=max_allowed_sentence_len
    )
    if torch.cuda.is_available():
        loss_function = loss_function.cuda()
        model = model.cuda()

    optimizer = torch.optim.Adam(list(model.parameters()), learning_rate)


    # Train
    train(model, num_epochs, loss_function, optimizer, training_set.target_vocab_size, force, iterations, save_dir="./")

