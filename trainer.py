import os 

import numpy as np
import torch as t

from sklearn.metrics import f1_score


CHECKPOINTS_DIR = "./checkpoints"

class Trainer:

    def __init__(self,
                 model,                        # Model to be trained
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 scheduler=None,               # Schedule for learning rate decay
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True):                   # Whether to use the GPU
                 
        self._model = model
        self._crit = crit
        self._optim = optim
        self._scheduler = scheduler
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        if not os.path.exists(CHECKPOINTS_DIR):
            os.mkdir(CHECKPOINTS_DIR)
        t.save({'state_dict': self._model.state_dict()}, CHECKPOINTS_DIR + '/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load(CHECKPOINTS_DIR + '/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # Model being run
              x,                         # Model input (or a tuple for multiple inputs)
              fn,                        # Where to save the model (can be a file or file-like object)
              export_params=True,        # Store the trained parameter weights inside the model file
              opset_version=11,          # The ONNX version to export the model to
              do_constant_folding=True,  # Whether to execute constant folding for optimization
              input_names = ['input'],   # The model's input names
              output_names = ['output'], # The model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # Variable lenght axes
                            'output' : {0 : 'batch_size'}})
            

    def train_step(self, x, y):
        # Reset gradients
        self._optim.zero_grad()

        # Propagate through the network and calculate loss
        y_pred = self._model(x)
        loss = self._crit(y_pred, y)
        
        # Compute gradient by backward propagation and update weights
        loss.backward()
        self._optim.step()

        return loss
        
    
    def val_test_step(self, x, y):
        # Propagate through the network and calculate loss
        y_pred = self._model(x)
        l = self._crit(y_pred, y)
        return l, y_pred
        
        
    def train_epoch(self):
        self._train_dl.mode = "train"
        self._model.train()

        total_loss = 0

        # Iterate through the training set and compute total loss
        for x, y in self._train_dl:
            x, y = (x.cuda(), y.cuda()) if self._cuda else (x, y)
            total_loss += self.train_step(x, y).item()
        
        for s in self._scheduler:
            s.step()
        
        # Calculate the average loss for the epoch and return it
        avg_loss = total_loss / len(self._train_dl)
        print("TRAIN loss: ", avg_loss)

        return avg_loss
    
    
    def calculate_metrics(self, total_loss, y_preds, y_grounds):
        avg_loss = total_loss / len(self._val_test_dl)

        f1_crack = f1_score(y_true=y_grounds[:, 0, 0], y_pred=y_preds[:, 0, 0], average='binary')
        f1_inactive = f1_score(y_true=y_grounds[:, 0, 1], y_pred=y_preds[:, 0, 1], average='binary')
        f1_mean = (f1_crack + f1_inactive) / 2

        # print("#Cracks: ", len(y_preds[y_preds[:, 0, 0] == 1]))
        # print("#Inactive: ", len(y_preds[y_preds[:, 0, 1] == 1]))
        # print("#Both: ", len(y_preds[(y_preds[:, 0, 0] == 1) & (y_preds[:, 0, 1] == 1)]))

        print("VAL loss: ", avg_loss)
        # print("\nF1 crack: ", f1_crack)
        # print("F1 inactive: ", f1_inactive)
        print("F1 mean: ", f1_mean, "\n")

        return avg_loss, f1_mean


    @t.no_grad()
    def val_test(self):
        self._val_test_dl.mode = "val"
        self._model.eval()
        
        y_preds = []
        y_grounds = []

        total_loss = 0

        # Iterate through the validation set
        for x, y in self._val_test_dl:
            x, y = (x.cuda(), y.cuda()) if self._cuda else (x, y)
            l, y_pred = self.val_test_step(x, y)
            total_loss += l.item()

            # Save the predictions and the labels for each batch
            y_preds.append(np.around(y_pred.cpu().numpy()))
            y_grounds.append(y.cpu().numpy())
        

        # Calculate relevant metrics and return them
        return self.calculate_metrics(total_loss, np.array(y_preds), np.array(y_grounds))

    
    def fit(self, epochs=-1):
        assert epochs > 0
        
        # Save average train and validation and f1-score for each epoch
        train_losses = []
        val_losses = []
        val_f1 = []

        epoch = 0

        while True:
            
            if epoch == epochs:
                break

            print("--- Epoch", epoch, "---")
            train_losses.append(self.train_epoch())
            avg_loss, f1_mean = self.val_test()
            val_losses.append(avg_loss)
            val_f1.append(f1_mean)

            # Save model if it reaches a certain f1-score
            if f1_mean > 0.6:
                self.save_checkpoint(epoch)

            epoch += 1
        
        return train_losses, val_losses, val_f1