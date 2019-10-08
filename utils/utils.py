from visdom import Visdom
import torch

viz = Visdom()


class LambdaLR:
    def __init__(self, n_epoch, offset, decay_epoch):
        self.n_epoch = n_epoch
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        factor = pow(0.5, int((self.offset + epoch) / self.decay_epoch))
        return factor


def create_vis_plot(_xlabel, _ylabel, _title):
    return viz.line(
        X=torch.zeros((1,)),
        Y=torch.zeros((1,)),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title
        )
    )


def update_vis_plot(iteration, loss, window1, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 1))*iteration,
        Y=torch.ones((1, 1))*loss,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


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