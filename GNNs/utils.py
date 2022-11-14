import dgl
import torch
import numpy as np


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    loop_graphs = [dgl.add_self_loop(graph) for graph in graphs]
    return dgl.batch(loop_graphs), torch.tensor(labels, dtype=torch.long)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, checkpoint_file_path='./checkpoints/'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint_file_path = checkpoint_file_path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f'valid loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model to {self.checkpoint_file_path}...')
        torch.save(model.state_dict(), self.checkpoint_file_path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss
