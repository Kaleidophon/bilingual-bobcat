"""
Module defining the model training function.
"""


def train():
    # TODO: Implement
    ...


epoch_losses = []
iterations = len(data_loader)

for epoch in range(0, num_epochs):
    start = time.time()
    batch_losses = []
    batch = 0
    for source_batch, target_batch, source_lengths, target_lengths, batch_positions in data_loader:
        batch_start = time.time()
        max_len = source_lengths.max()
        source_batch = source_batch[:, :max_len]
        batch_positions = batch_positions[:, :max_len]

        source_batch = torch.autograd.Variable(source_batch)
        target_batch = torch.autograd.Variable(target_batch)
        source_lengths = torch.autograd.Variable(source_lengths)
        target_lengths = torch.autograd.Variable(target_lengths)
        batch_positions = torch.autograd.Variable(batch_positions)

        if (cuda):
            source_batch = source_batch.cuda()
            target_batch = target_batch.cuda()
            source_lengths = source_lengths.cuda()
            target_lengths = target_lengths.cuda()
            batch_positions = batch_positions.cuda()

        # get encoder and decoder outputs
        encoder_out = encoder(source_batch, batch_positions)
        decoder_out = decoder(target_batch, target_lengths, source_lengths, encoder_out, max_len)

        # remove the start token from the targets and the end token from the decoder
        decoder_out2 = decoder_out[:, :-1, :]
        target_batch2 = target_batch[:, 1:decoder_out.size(1)]

        # pytorch expects the inputs for the loss function to be: (batch x classes)
        # cant handle sentences so we just transform our data to treath each individual word as a batch:
        # so new batch size = old batch * padded sentence length
        # should not matter as it all gets averaged out anyway
        decoder_out3 = decoder_out2.contiguous().view(decoder_out2.size(0) * decoder_out2.size(1), decoder_out2.size(2))
        target_batch3 = target_batch2.contiguous().view(target_batch2.size(0) * target_batch2.size(1))

        # calculate loss
        loss = loss_function(decoder_out3, target_batch3)
        batch_losses.append(loss)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - batch_start
        print('\r[Epoch {:03d}/{:03d}] Batch {:06d}/{:06d} [{:.1f}/s] '.format(epoch + 1, num_epochs, batch + 1,
                                                                               iterations, batch_time), end='')
        batch += 1

    avg_loss = sum(batch_losses) / iterations
    epoch_losses.append(avg_loss)
    print('Time: {:.1f}s Loss: {:.3f} score2?: {:.6f}'.format(time.time() - start, avg_loss, 0))