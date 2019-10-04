from visdom import Visdom
import torch

viz = Visdom()


class LambdaLR:
    def __init__(self, n_epoch, offset, total_batch_size, decay_batch_size):
        self.n_epoch = n_epoch
        self.offset = offset
        self.total_batch_size = total_batch_size
        self.decay_batch_size = decay_batch_size

    def step(self, epoch):
        factor = pow(0.5, int(((self.offset + epoch) * self.total_batch_size) / self.decay_batch_size))
        return factor


def create_vis_plot(_xlabel, _ylabel, _title):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title
        )
    )


def update_vis_plot(iteration, loss, window1, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3))*iteration,
        Y=torch.ones((1, 3))*loss,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count