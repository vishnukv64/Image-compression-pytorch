import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--in_channels', type=int, default=3, help='the number of input channels')
parser.add_argument('--out_channels', type=int, default=3, help='the number of output channels')
parser.add_argument('--storing_channels', type=int, default=3, help='the number of channels to store')
parser.add_argument('--nf', type=int, default=32, help='the number of channels in models')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--b1', type=float, default=0.9, help='coefficients used for computing running averages of gradient and its square')
parser.add_argument('--b2', type=float, default=0.999, help='coefficients used for computing running averages of gradient and its square')
parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay')
parser.add_argument('--decay_batch_size', type=int, default=1000, help='decayed batch size')
parser.add_argument('--perceptual_loss_factor', type=float, default=0.003)
parser.add_argument('--content_loss_factor', type=float, default=1.0)
parser.add_argument('--generator_loss_factor', type=float, default=0.0001)
parser.add_argument('--penalty_loss_factor', type=float, default=10.0)
parser.add_argument('--discriminator_loss_factor', type=float, default=1.0)
parser.add_argument('--eta', type=float, default=0.3, help='image interpolation factor for penalty in wgan')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory for saving parameter')
parser.add_argument('--epoch', type=int, default=0, help='start epoch')
parser.add_argument('--num_epoch', type=int, default=100, help='end epoch')


def get_config():
    return parser.parse_args()