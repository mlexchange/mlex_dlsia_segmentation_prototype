import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from dlsia.core.train_scripts import segmentation_metrics
import logging
from dvclive import Live

def custom_collate(batch):
    elem = batch[0]
    #print(f'elem type: {type(elem)}')
    first_data = elem[0]
    #print(f'first_data_size: {first_data.shape}')
    if isinstance(elem, tuple) and elem[0].ndim == 4:
        data, mask = zip(*batch)
        concated_data = torch.cat(data, dim=0) # concat on the first dim without introducing another dim -> keep in the 4d realm
        concated_mask = torch.cat(mask, dim=0)
        #print(f'concated_data shape: {concated_data.shape}')
        #print(f'concated_mask shape: {concated_mask.shape}')
        return concated_data, concated_mask
    elif isinstance(elem, torch.Tensor) and elem.ndim == 4:
        #print(f'batch size: {len(batch)}')
        concated_data = torch.cat(batch, dim=0) # concat on the first dim without introducing another dim -> keep in the 4d realm
        #print(f'concated_data shape: {concated_data.shape}')
        return concated_data
    else:  # Fall back to `default_collate` as suggested by PyTorch documentation
        return default_collate(batch)

# Train Val Split
def train_val_split(dataset, parameters):
    '''
    This funnction splits the given tiled_dataset object into the train set and val set using torch's built in random_split function.

    Caution: the random_split does not taken class balance into account. Future upgrades for that direction would requrie sampler from torch.
    '''

    # Set Dataloader parameters (Note: we randomly shuffle the training set upon each pass)
    train_loader_params = {'batch_size': parameters.batch_size_train,
                        'shuffle': parameters.shuffle_train}
    val_loader_params = {'batch_size': parameters.batch_size_val,
                        'shuffle': parameters.shuffle_val}

    # Build Dataloaders
    val_pct = parameters.val_pct
    val_size = int(val_pct*len(dataset))
    #print(f'length of dataset: {len(dataset)}')
    #print(f'length of val_size: {val_size}')
    if len(dataset) == 1:
        train_loader = DataLoader(dataset, **train_loader_params, collate_fn=custom_collate)
        val_loader = None
    elif val_size == 0:
        train_size = len(dataset) - 1
        train_data, val_data = random_split(dataset, [train_size, 1])
        #print(f'train_data size: {len(train_data)}')
        train_loader = DataLoader(train_data, **train_loader_params, collate_fn=custom_collate)
        val_loader = DataLoader(val_data, **val_loader_params, collate_fn=custom_collate)
    else:
        train_size = len(dataset) - val_size
        train_data, val_data = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_data, **train_loader_params, collate_fn=custom_collate)
        val_loader = DataLoader(val_data, **val_loader_params, collate_fn=custom_collate)
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
                'epoch': [epoch],
                'loss': [loss], 
                'val_loss': [val_loss], 
                'F1_micro': [F1_micro], 
                'F1_macro': [F1_macro],
                'F1_val_micro': [F1_val_micro],
                'F1_val_macro': [F1_val_macro]
            }
            )
    
    else:
        table = pd.DataFrame({
                'epoch': [epoch],
                'loss': [loss], 
                'F1_micro': [F1_micro], 
                'F1_macro': [F1_macro]})

    return table


# Segmentation
def segment(net, device, inference_loader, qlty_object):
    net.to(device)   # send network to GPU
    patch_preds = [] # store results for patches
    for idx, batch in enumerate(inference_loader):
        print(f'{idx}/{len(inference_loader)}')
        with torch.no_grad():
            # Necessary data recasting
            batch = batch.type(torch.FloatTensor)
            batch = batch.to(device)
            # Input passed through networks here
            output_network = net(batch)
            patch_preds.append(output_network)
    
    patch_preds = torch.concat(patch_preds)
    stitched_result, weights = qlty_object.stitch(patch_preds)
    # Individual output passed through argmax to get predictions
    seg = torch.argmax(stitched_result.cpu(), dim=1).numpy()
    return seg



## 20240304, added by xchong ##
class Trainer():
    def __init__(self, net, trainloader, validationloader, NUM_EPOCHS,
                       criterion, optimizer, device, dvclive=None,
                       savepath=None, saveevery=None,
                       scheduler=None, show=0,
                       use_amp=False, clip_value=None):


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
        :param dvclive: use dvclive object to save metrics
        :param savepath: filepath in which we save networks intermittently
        :param saveevery: integer n for saving network every n epochs
        :param scheduler: an optional schedular. can be None
        :param show: print stats every n-th epoch
        :param use_amp: use pytorch automatic mixed precision
        :param clip_value: value for gradient clipping. Can be None.
        :return: A network and run summary stats
        """

        self.net = net
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.NUM_EPOCHS = NUM_EPOCHS
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.dvclive = dvclive
        self.savepath = savepath
        self.saveevery = saveevery
        self.scheduler = scheduler
        self.show = show
        self.use_amp = use_amp
        self.clip_value = clip_value

        self.train_loss = []
        self.F1_train_trace_micro = []
        self.F1_train_trace_macro = []

          

        # Skip validation steps if False or None loaded
        if self.validationloader is False:
            self.validationloader = None
        if self.validationloader is not None:
            self.validation_loss = []
            self.F1_validation_trace_micro = []
            self.F1_validation_trace_macro = []

        self.best_score = 1e10
        self.best_index = 0
        self.best_state_dict = None

        if self.savepath is not None:
            if self.saveevery is None:
                self.saveevery = 1


    def train_one_epoch(self,epoch):
        running_train_loss = 0.0
        running_F1_train_micro = 0.0
        running_F1_train_macro = 0.0
        tot_train = 0.0

        if self.validationloader is not None:
            running_validation_loss = 0.0
            running_F1_validation_micro = 0.0
            running_F1_validation_macro = 0.0
            tot_val = 0.0
        count = 0

        for data in self.trainloader:
            count += 1
            noisy, target = data  # load noisy and target images
            N_train = noisy.shape[0]
            tot_train += N_train

            noisy = noisy.type(torch.FloatTensor)
            target = target.type(torch.LongTensor)
            noisy = noisy.to(self.device)
            target = target.to(self.device)

            if self.criterion.__class__.__name__ == 'CrossEntropyLoss':
                target = target.type(torch.LongTensor)
                target = target.to(self.device).squeeze(1)

            if self.use_amp is False:
                # forward pass, compute loss and accuracy
                output = self.net(noisy)
                loss = self.criterion(output, target)

                # backpropagation
                self.optimizer.zero_grad()
                loss.backward()
            else:
                scaler = torch.cuda.amp.GradScaler()
                with torch.cuda.amp.autocast():
                    # forward pass, compute loss and accuracy
                    output = self.net(noisy)
                    loss = self.criterion(output, target)

                # backpropagation
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()

                # update the parameters
                scaler.step(self.optimizer)
                scaler.update()

            # update the parameters
            if self.clip_value is not None:
                torch.nn.utils.clip_grad_value_(self.net.parameters(), self.clip_value)
            self.optimizer.step()


            tmp_micro, tmp_macro = segmentation_metrics(output, target)

            running_F1_train_micro += tmp_micro.item()
            running_F1_train_macro += tmp_macro.item()
            running_train_loss += loss.item()
        if self.scheduler is not None:
            self.scheduler.step()

        # compute validation step
        if self.validationloader is not None:
            with torch.no_grad():
                for x, y in self.validationloader:
                    x = x.type(torch.FloatTensor)
                    y = y.type(torch.LongTensor)
                    x = x.to(self.device)
                    y = y.to(self.device)
                    N_val = y.shape[0]
                    tot_val += N_val
                    if self.criterion.__class__.__name__ == 'CrossEntropyLoss':
                        y = y.type(torch.LongTensor)
                        y = y.to(self.device).squeeze(1)

                    # forward pass, compute validation loss and accuracy
                    if self.use_amp is False:
                        yhat = self.net(x)
                        val_loss = self.criterion(yhat, y)
                    else:
                        with torch.cuda.amp.autocast():
                            yhat = self.net(x)
                            val_loss = self.criterion(yhat, y)

                    tmp_micro, tmp_macro = segmentation_metrics(yhat, y)
                    running_F1_validation_micro += tmp_micro.item()
                    running_F1_validation_macro += tmp_macro.item()

                    # update running validation loss and accuracy
                    running_validation_loss += val_loss.item()

        loss = running_train_loss / len(self.trainloader)
        F1_micro = running_F1_train_micro / len(self.trainloader)
        F1_macro = running_F1_train_macro / len(self.trainloader)
        self.train_loss.append(loss)
        self.F1_train_trace_micro.append(F1_micro)
        self.F1_train_trace_macro.append(F1_macro)


        if self.validationloader is not None:
            val_loss = running_validation_loss / len(self.validationloader)
            F1_val_micro = running_F1_validation_micro / len(self.validationloader)
            F1_val_macro = running_F1_validation_macro / len(self.validationloader)
            self.validation_loss.append(val_loss)
            self.F1_validation_trace_micro.append(F1_val_micro)
            self.F1_validation_trace_macro.append(F1_val_macro)



            # Update loss parquet file
            if epoch == 0:
                self.table = pd.DataFrame(
                {
                    'epoch': [epoch],
                    'loss': [loss],
                    'val_loss': [val_loss],
                    'F1_micro': [F1_micro],
                    'F1_macro': [F1_macro],
                    'F1_val_micro': [F1_val_micro],
                    'F1_val_macro': [F1_val_macro]
                }
                )
            else:
                self.table = pd.concat([
                self.table,
                pd.DataFrame(
                    {
                    'epoch': [epoch],
                    'loss': [loss],
                    'val_loss': [val_loss],
                    'F1_micro': [F1_micro],
                    'F1_macro': [F1_macro],
                    'F1_val_micro': [F1_val_micro],
                    'F1_val_macro': [F1_val_macro]
                    }
                )
                ])
        else:


            # Update loss parquet file
            if epoch == 0:
                self.table = pd.DataFrame({
                        'epoch': [epoch],
                        'loss': [loss],
                        'F1_micro': [F1_micro],
                        'F1_macro': [F1_macro]})
            else:
                self.table = pd.concat([
                self.table,
                pd.DataFrame(
                        {
                        'epoch': [epoch],
                        'loss': [loss],
                        'F1_micro': [F1_micro],
                        'F1_macro': [F1_macro]
                        }
                )
                ])
        self.table.to_parquet(self.savepath+'/losses_per_epoch.parquet', engine='pyarrow')

        if self.show != 0:
            learning_rates = []
            for param_group in self.optimizer.param_groups:
                learning_rates.append(param_group['lr'])
            mean_learning_rate = np.mean(np.array(learning_rates))
            if np.mod(epoch + 1, self.show) == 0:
                if self.validationloader is not None:
                    print(
                              f'Epoch {epoch + 1} of {self.NUM_EPOCHS} | Learning rate {mean_learning_rate:4.3e}')
                    print(
                              f'   Training Loss: {loss:.4e} | Validation Loss: {val_loss:.4e}')
                    print(
                              f'   Micro Training F1: {F1_micro:.4f} | Micro Validation F1: {F1_val_micro:.4f}')
                    print(
                              f'   Macro Training F1: {F1_macro:.4f} | Macro Validation F1: {F1_val_macro:.4f}')
                else:
                    print(
                              f'Epoch {epoch + 1} of {self.NUM_EPOCHS} | Learning rate {mean_learning_rate:4.3e}')
                    print(
                              f'   Training Loss: {loss:.4e} | Micro Training F1: {F1_micro:.4f} | Macro Training F1: {F1_macro:.4f}')

        if self.validationloader is not None:
            if val_loss < self.best_score:
                self.best_state_dict = self.net.state_dict()
                self.best_index = epoch
                self.best_score = val_loss
        else:
            if loss < self.best_score:
                self.best_state_dict = self.net.state_dict()
                self.best_index = epoch
                self.best_score = loss

            if self.savepath is not None:
                torch.save(self.best_state_dict, self.savepath + '/net_best')
                print('   Best network found and saved')
                print('')

        if self.savepath is not None:
            if np.mod(epoch + 1, self.saveevery) == 0:
                torch.save(self.net.state_dict(), self.savepath + '/net_checkpoint')
                print('   Network intermittently saved')
                print('')

        return True

    def train_segmentation(self):

        for epoch in range(self.NUM_EPOCHS):

            ## 20240304, modified by xchong ##
            self.train_one_epoch(epoch)

            if self.dvclive is not None:
                self.dvclive.log_metric("train/loss", self.train_loss[-1])
                self.dvclive.log_metric("train/F1_micro", self.F1_train_trace_micro[-1])
                self.dvclive.log_metric("train/F1_macro", self.F1_train_trace_macro[-1])

                if self.validationloader is not None:
                    self.dvclive.log_metric("val/loss", self.validation_loss[-1])
                    self.dvclive.log_metric("val/F1_micro", self.F1_validation_trace_micro[-1])
                    self.dvclive.log_metric("val/F1_macro", self.F1_validation_trace_macro[-1])
                self.dvclive.next_step()
            ## 20240304, modified by xchong ##

        if self.validationloader is None:
            self.validation_loss = None
            self.F1_validation_trace_micro = None
            self.F1_validation_trace_macro = None

        results = {"Training loss": self.train_loss,
                         "Validation loss": self.validation_loss,
                         "F1 training micro": self.F1_train_trace_micro,
                         "F1 training macro": self.F1_train_trace_macro,
                         "F1 validation micro": self.F1_validation_trace_micro,
                         "F1 validation macro": self.F1_validation_trace_macro,
                         "Best model index": self.best_index}

        self.net.load_state_dict(self.best_state_dict)
        return self.net, results
## 20240304, added by xchong ##
