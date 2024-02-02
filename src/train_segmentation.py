import logging

import numpy as np
import torch

from dlsia.core.train_scripts import segmentation_metrics
from save_loss import save_loss


def train_segmentation(
        net,
        trainloader,
        validationloader,
        NUM_EPOCHS,
        criterion,
        optimizer,
        device,
        savepath=None,
        saveevery=None,
        scheduler=None,
        show=0,
        use_amp=False,
        clip_value=None
        ):
    """
    Loop through epochs passing images to be segmented on a pixel-by-pixel
    basis.

    :param net: input network
    :param trainloader: data loader with training data
    :param validationloader: data loader with validation data
    :param NUM_EPOCHS: number of epochs
    :param criterion: target function
    :param optimizer: optimization engine
    :param device: the device where we calculate things
    :param savepath: filepath in which we save networks intermittently
    :param saveevery: integer n for saving network every n epochs
    :param scheduler: an optional schedular. can be None
    :param show: print stats every n-th epoch
    :param use_amp: use pytorch automatic mixed precision
    :param clip_value: value for gradient clipping. Can be None.
    :return: A network and run summary stats
    """

    train_loss = []
    F1_train_trace_micro = []
    F1_train_trace_macro = []

    # Skip validation steps if False or None loaded
    if validationloader is False:
        validationloader = None
    if validationloader is not None:
        validation_loss = []
        F1_validation_trace_micro = []
        F1_validation_trace_macro = []

    best_score = 1e10
    best_index = 0
    best_state_dict = None

    if savepath is not None:
        if saveevery is None:
            saveevery = 1

    for epoch in range(NUM_EPOCHS):
        running_train_loss = 0.0
        running_F1_train_micro = 0.0
        running_F1_train_macro = 0.0
        tot_train = 0.0

        if validationloader is not None:
            running_validation_loss = 0.0
            running_F1_validation_micro = 0.0
            running_F1_validation_macro = 0.0
            tot_val = 0.0
        count = 0

        for data in trainloader:
            count += 1
            noisy, target = data  # load noisy and target images
            N_train = noisy.shape[0]
            tot_train += N_train

            noisy = noisy.type(torch.FloatTensor)
            target = target.type(torch.LongTensor)
            noisy = noisy.to(device)
            target = target.to(device)

            if criterion.__class__.__name__ == 'CrossEntropyLoss':
                target = target.type(torch.LongTensor)
                target = target.to(device).squeeze(1)

            if use_amp is False:
                # forward pass, compute loss and accuracy
                output = net(noisy)
                loss = criterion(output, target)

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
            else:
                scaler = torch.cuda.amp.GradScaler()
                with torch.cuda.amp.autocast():
                    # forward pass, compute loss and accuracy
                    output = net(noisy)
                    loss = criterion(output, target)

                # backpropagation
                optimizer.zero_grad()
                scaler.scale(loss).backward()

                # update the parameters
                scaler.step(optimizer)
                scaler.update()

            # update the parameters
            if clip_value is not None:
                torch.nn.utils.clip_grad_value_(net.parameters(), clip_value)
            optimizer.step()


            tmp_micro, tmp_macro = segmentation_metrics(output, target)

            running_F1_train_micro += tmp_micro.item()
            running_F1_train_macro += tmp_macro.item()
            running_train_loss += loss.item()
        if scheduler is not None:
            scheduler.step()

        # compute validation step
        if validationloader is not None:
            with torch.no_grad():
                for x, y in validationloader:
                    x = x.to(device)
                    y = y.to(device)
                    N_val = y.shape[0]
                    tot_val += N_val
                    if criterion.__class__.__name__ == 'CrossEntropyLoss':
                        y = y.type(torch.LongTensor)
                        y = y.to(device).squeeze(1)

                    # forward pass, compute validation loss and accuracy
                    if use_amp is False:
                        yhat = net(x)
                        val_loss = criterion(yhat, y)
                    else:
                        with torch.cuda.amp.autocast():
                            yhat = net(x)
                            val_loss = criterion(yhat, y)

                    tmp_micro, tmp_macro = segmentation_metrics(yhat, y)
                    running_F1_validation_micro += tmp_micro.item()
                    running_F1_validation_macro += tmp_macro.item()

                    # update running validation loss and accuracy
                    running_validation_loss += val_loss.item()

        loss = running_train_loss / len(trainloader)
        F1_micro = running_F1_train_micro / len(trainloader)
        F1_macro = running_F1_train_macro / len(trainloader)
        train_loss.append(loss)
        F1_train_trace_micro.append(F1_micro)
        F1_train_trace_macro.append(F1_macro)

        if validationloader is not None:
            val_loss = running_validation_loss / len(validationloader)
            F1_val_micro = running_F1_validation_micro / len(validationloader)
            F1_val_macro = running_F1_validation_macro / len(validationloader)
            validation_loss.append(val_loss)
            F1_validation_trace_micro.append(F1_val_micro)
            F1_validation_trace_macro.append(F1_val_macro)
        
        save_loss(
            validationloader,
            savepath,
            epoch,
            loss,
            F1_micro,
            F1_macro,
            val_loss=val_loss,
            F1_val_micro=F1_val_micro,
            F1_val_macro=F1_val_macro,
            )

        if show != 0:
            learning_rates = []
            for param_group in optimizer.param_groups:
                learning_rates.append(param_group['lr'])
            mean_learning_rate = np.mean(np.array(learning_rates))
            if np.mod(epoch + 1, show) == 0:
                if validationloader is not None:
                    logging.INFO(
                        f'Epoch {epoch + 1} of {NUM_EPOCHS} | Learning rate {mean_learning_rate:4.3e}')
                    logging.INFO(
                        f'   Training Loss: {loss:.4e} | Validation Loss: {val_loss:.4e}')
                    logging.INFO(
                        f'   Micro Training F1: {F1_micro:.4f} | Micro Validation F1: {F1_val_micro:.4f}')
                    logging.INFO(
                        f'   Macro Training F1: {F1_macro:.4f} | Macro Validation F1: {F1_val_macro:.4f}')
                else:
                    logging.INFO(
                        f'Epoch {epoch + 1} of {NUM_EPOCHS} | Learning rate {mean_learning_rate:4.3e}')
                    logging.INFO(
                        f'   Training Loss: {loss:.4e} | Micro Training F1: {F1_micro:.4f} | Macro Training F1: {F1_macro:.4f}')

        if validationloader is not None:
            if val_loss < best_score:
                best_state_dict = net.state_dict()
                best_index = epoch
                best_score = val_loss
        else:
            if loss < best_score:
                best_state_dict = net.state_dict()
                best_index = epoch
                best_score = loss

            if savepath is not None:
                torch.save(best_state_dict, savepath + '/net_best')
                logging.INFO('   Best network found and saved')
                logging.INFO('')

        if savepath is not None:
            if np.mod(epoch + 1, saveevery) == 0:
                torch.save(net.state_dict(), savepath + '/net_checkpoint')
                logging.INFO('   Network intermittently saved')
                logging.INFO('')

    if validationloader is None:
        validation_loss = None
        F1_validation_trace_micro = None
        F1_validation_trace_macro = None

    results = {"Training loss": train_loss,
               "Validation loss": validation_loss,
               "F1 training micro": F1_train_trace_micro,
               "F1 training macro": F1_train_trace_macro,
               "F1 validation micro": F1_validation_trace_micro,
               "F1 validation macro": F1_validation_trace_macro,
               "Best model index": best_index}

    net.load_state_dict(best_state_dict)
    return net, results