import logging

import numpy as np
import pandas as pd
import torch
from dlsia.core.train_scripts import segmentation_metrics
from dvclive import Live
from qlty import cleanup
from qlty.qlty2D import NCYXQuilt
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.dataloader import default_collate


def custom_collate(batch):
    elem = batch[0]
    if isinstance(elem, tuple) and elem[0].ndim == 4:
        data, mask = zip(*batch)
        concated_data = torch.cat(data, dim=0)
        # concat on the first dim without introducing another dim -> keep in the 4d realm
        concated_mask = torch.cat(mask, dim=0)
        return concated_data, concated_mask
    elif isinstance(elem, torch.Tensor) and elem.ndim == 4:
        # concat on the first dim without introducing another dim -> keep in the 4d realm
        concated_data = torch.cat(batch, dim=0)
        return concated_data
    else:  # Fall back to `default_collate` as suggested by PyTorch documentation
        return default_collate(batch)


# Train Val Split
def train_val_split(dataset, parameters):
    """
    This funnction splits the given tiled_dataset object
    into the train set and val set using torch's built in random_split function.


    Caution: the random_split does not taken class balance into account.
    Future upgrades for that direction would requrie sampler from torch.
    """

    # Set Dataloader parameters (Note: we randomly shuffle the training set upon each pass)
    train_loader_params = {
        "batch_size": parameters.batch_size_train,
        "shuffle": parameters.shuffle_train,
    }
    val_loader_params = {"batch_size": parameters.batch_size_val, "shuffle": False}

    # Build Dataloaders
    val_pct = parameters.val_pct
    val_size = max(int(val_pct * len(dataset)), 1) if len(dataset) > 1 else 0
    if val_size == 0:
        train_loader = DataLoader(
            dataset, **train_loader_params, collate_fn=custom_collate
        )
        val_loader = None
    else:
        train_size = len(dataset) - val_size
        train_data, val_data = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(
            train_data, **train_loader_params, collate_fn=custom_collate
        )
        val_loader = DataLoader(
            val_data, **val_loader_params, collate_fn=custom_collate
        )
    return train_loader, val_loader


def crop_split_load(images, masks, parameters, qlty_border_weight=0.2):
    # Normalize by clipping to 1% and 99% percentiles
    low = np.percentile(images.ravel(), 1)
    high = np.percentile(images.ravel(), 99)
    images = np.clip((images - low) / (high - low), 0, 1)

    images = torch.from_numpy(images)
    masks = torch.from_numpy(masks)

    qlty_window = parameters.qlty_window
    qlty_step = parameters.qlty_step
    qlty_border = parameters.qlty_border

    if images.ndim == 3:
        images = images.unsqueeze(1)
    elif images.ndim == 2:
        images = images.unsqueeze(0).unsqueeze(0)

    if masks.ndim == 2:
        masks = masks.unsqueeze(0)

    qlty_object = NCYXQuilt(
        X=images.shape[-1],
        Y=images.shape[-2],
        window=(qlty_window, qlty_window),
        step=(qlty_step, qlty_step),
        border=(qlty_border, qlty_border),
        border_weight=qlty_border_weight,
    )
    patched_images, patched_masks = qlty_object.unstitch_data_pair(images, masks)
    # Clean up unlabeled patches

    patched_images, patched_masks, _ = (
        cleanup.weed_sparse_classification_training_pairs_2D(
            patched_images,
            patched_masks,
            missing_label=-1,
            border_tensor=qlty_object.border_tensor(),
        )
    )
    dataset = TensorDataset(patched_images, patched_masks)
    # Set Dataloader parameters (Note: we randomly shuffle the training set upon each pass)
    train_loader_params = {
        "batch_size": parameters.batch_size_train,
        "shuffle": parameters.shuffle_train,
    }
    val_loader_params = {"batch_size": parameters.batch_size_val, "shuffle": False}

    val_pct = parameters.val_pct
    val_size = max(int(val_pct * len(dataset)), 1) if len(dataset) > 1 else 0
    if val_size == 0:
        train_loader = DataLoader(dataset, **train_loader_params)
        val_loader = None
    else:
        train_size = len(dataset) - val_size
        train_data, val_data = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_data, **train_loader_params)
        val_loader = DataLoader(val_data, **val_loader_params)
    return train_loader, val_loader


# Save Loss
def save_loss(
    validationloader,
    savepath,
    epoch,
    loss,
    F1_micro,
    F1_macro,
    val_loss=None,
    F1_val_micro=None,
    F1_val_macro=None,
):
    if validationloader is not None:
        table = pd.DataFrame(
            {
                "epoch": [epoch],
                "loss": [loss],
                "val_loss": [val_loss],
                "F1_micro": [F1_micro],
                "F1_macro": [F1_macro],
                "F1_val_micro": [F1_val_micro],
                "F1_val_macro": [F1_val_macro],
            }
        )

    else:
        table = pd.DataFrame(
            {
                "epoch": [epoch],
                "loss": [loss],
                "F1_micro": [F1_micro],
                "F1_macro": [F1_macro],
            }
        )

    return table


# Train models
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
    clip_value=None,
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

    best_score = 0  # will be set in the first epoch
    best_index = 0
    best_state_dict = None

    if savepath is not None:
        if saveevery is None:
            saveevery = 1

    losses = pd.DataFrame()
    with Live(savepath, report="html") as live:
        for epoch in range(NUM_EPOCHS):
            print(
                f"*****  memory allocated at epoch {epoch} is {torch.cuda.memory_allocated(0)}"
            )
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

                if criterion.__class__.__name__ == "CrossEntropyLoss":
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
                        x = x.type(torch.FloatTensor)
                        y = y.type(torch.LongTensor)
                        x = x.to(device)
                        y = y.to(device)
                        N_val = y.shape[0]
                        tot_val += N_val
                        if criterion.__class__.__name__ == "CrossEntropyLoss":
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
            val_loss = None
            if validationloader is not None:
                val_loss = running_validation_loss / len(validationloader)
                F1_val_micro = running_F1_validation_micro / len(validationloader)
                F1_val_macro = running_F1_validation_macro / len(validationloader)
                validation_loss.append(val_loss)
                F1_validation_trace_micro.append(F1_val_micro)
                F1_validation_trace_macro.append(F1_val_macro)

                live.log_metric("train/nes", loss)
                live.log_metric("train/F1_micro", F1_micro)
                live.log_metric("train/F1_macro", F1_macro)
                live.log_metric("val/loss", val_loss)
                live.log_metric("val/F1_micro", F1_val_micro)
                live.log_metric("val/F1_macro", F1_val_macro)
                live.next_step()

            print(f"Epoch: {epoch}")

            # Note: This is a very temporary solution to address the single frame mask case.
            if validationloader is None:
                F1_val_micro = None
                F1_val_macro = None

            table = save_loss(
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

            losses = pd.concat([losses, table])

            if show != 0:
                learning_rates = []
                for param_group in optimizer.param_groups:
                    learning_rates.append(param_group["lr"])
                mean_learning_rate = np.mean(np.array(learning_rates))
                if np.mod(epoch + 1, show) == 0:
                    if validationloader is not None:
                        logging.info(
                            f"Epoch {epoch + 1} of {NUM_EPOCHS} | Learning rate {mean_learning_rate:4.3e}"
                        )
                        logging.info(
                            f"   Training Loss: {loss:.4e} | Validation Loss: {val_loss:.4e}"
                        )
                        logging.info(
                            f"   Micro Training F1: {F1_micro:.4f} | Micro Validation F1: {F1_val_micro:.4f}"
                        )
                        logging.info(
                            f"   Macro Training F1: {F1_macro:.4f} | Macro Validation F1: {F1_val_macro:.4f}"
                        )
                    else:
                        logging.info(
                            f"Epoch {epoch + 1} of {NUM_EPOCHS} | Learning rate {mean_learning_rate:4.3e}"
                        )
                        logging.info(
                            f"   Training Loss: {loss:.4e} | Micro Training F1: {F1_micro:.4f} "
                            + "| Macro Training F1: {F1_macro:.4f}"
                        )

            if epoch == 0:
                best_state_dict = net.state_dict()
                if validationloader is not None:
                    best_score = val_loss
                else:
                    best_score = loss

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

                    torch.save(best_state_dict, savepath + "/net_best")
                    logging.info("Best network found and saved")
                    logging.info("")

            if savepath is not None:
                if np.mod(epoch + 1, saveevery) == 0:
                    torch.save(net.state_dict(), savepath + "/net_checkpoint")
                    logging.info("Network intermittently saved")
                    logging.info("")

    if validationloader is None:
        validation_loss = []
        F1_validation_trace_micro = []
        F1_validation_trace_macro = []

    results = {
        "Training loss": train_loss,
        "Validation loss": validation_loss,
        "F1 training micro": F1_train_trace_micro,
        "F1 training macro": F1_train_trace_macro,
        "F1 validation micro": F1_validation_trace_micro,
        "F1 validation macro": F1_validation_trace_macro,
        "Best model index": best_index,
    }

    net.load_state_dict(best_state_dict)
    losses.to_parquet(savepath + "/losses.parquet", engine="pyarrow")
    return net, results


# Segmentation
def segment(net, device, inference_loader, qlty_object, tiled_client):
    net.to(device)  # send network to GPU
    frame_number = 0
    for idx, batch in enumerate(inference_loader):
        print(f"Batch: {idx+1}/{len(inference_loader)}")
        with torch.no_grad():
            # Necessary data recasting
            batch = batch.type(torch.FloatTensor)
            batch = batch.to(device)
            # Input passed through networks here
            patch_output = net(batch)
            patch_output = patch_output.cpu()
            stitched_result, weights = qlty_object.stitch(patch_output)
            # Individual output passed through argmax to get predictions
            seg_batch = torch.argmax(stitched_result, dim=1).numpy().astype(np.int8)
            batch_size = seg_batch.shape[0]
            for n in range(batch_size):
                frame = seg_batch[[n]]
                # Write back to Tiled for the single frame
                # TODO: Explore the use of threading to speed up this process.
                tiled_client.write_block(frame, block=(frame_number, 0, 0))
                print(f"Frame {frame_number+1} saved to Tiled")
                frame_number += 1

    return frame_number


def crop_seg_save(net, device, image, qlty_object, parameters, tiled_client, frame_idx):
    assert image.ndim == 2
    # Normalize by clipping to 1% and 99% percentiles
    low = np.percentile(image.ravel(), 1)
    high = np.percentile(image.ravel(), 99)
    image = np.clip((image - low) / (high - low), 0, 1)

    image = torch.from_numpy(image)
    image = image.unsqueeze(0).unsqueeze(0)

    softmax = torch.nn.Softmax(dim=1)
    # first patch up the image
    patches = qlty_object.unstitch(image)

    inference_loader = DataLoader(
        TensorDataset(patches),
        shuffle=False,
        batch_size=parameters.batch_size_inference,
    )

    net.eval().to(device)  # send network to GPU
    results = []
    for batch in inference_loader:
        with torch.no_grad():
            torch.cuda.empty_cache()
            patches = batch[0].type(torch.FloatTensor)
            tmp = softmax(net(patches.to(device))).cpu()
            results.append(tmp)
    results = torch.cat(results)
    stitched_result, _ = qlty_object.stitch(results)
    seg_result = torch.argmax(stitched_result, dim=1).numpy().astype(np.int8)
    # Write back to Tiled for the single frame
    # TODO: Explore the use of threading to speed up this process.
    tiled_client.write_block(seg_result, block=(frame_idx, 0, 0))
    print(f"Frame {frame_idx+1} result saved to Tiled")
    return seg_result
