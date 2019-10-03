import os
import paq
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torchvision.utils import save_image
from glob import glob
from visdom import Visdom

from model.encoder import Encoder
from model.decoder import Decoder
from model.discriminator import Discriminator
from utils.utils import LambdaLR, AverageMeter
from loss.loss import PerceptualLoss


class Trainer:
    def __init__(self, config, data_loader):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels

        self.nf = config.nf
        self.storing_channels = config.storing_channels

        self.lr = config.lr
        self.b1 = config.b1
        self.b2 = config.b2
        self.weight_decay = config.weight_decay
        self.decay_batch_size = config.decay_batch_size
        self.content_loss_factor = config.content_loss_factor
        self.perceptual_loss_factor = config.perceptual_loss_factor
        self.generator_loss_factor = config.generator_loss_factor
        self.discriminator_loss_factor = config.discriminator_loss_factor
        self.penalty_loss_factor = config.penalty_loss_factor
        self.eta = config.eta

        self.checkpoint_dir = config.checkpoint_dir
        self.sample_dir = config.sample_dir
        self.epoch = config.epoch
        self.num_epoch = config.num_epoch

        self.data_loader = data_loader

        self.content_losses = AverageMeter()
        self.generator_losses = AverageMeter()
        self.perceptual_losses = AverageMeter()
        self.discriminator_losses = AverageMeter()
        self.ae_losses = AverageMeter()

        self.visdom = Visdom()

        self.build_model()

    def train(self):
        optimizer_ae = Adam([self.Encoder.parameters(), self.Decoder.parameters()], self.lr, betas=(self.b1, self.b2),
                            weight_decay=self.weight_decay)
        optimizer_discriminator = Adam(self.Disciminator.parameters(), self.lr, betas=(self.b1, self.b2),
                                       weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_ae,
                                                         LambdaLR(self.num_epoch, self.epoch, len(self.data_loader),
                                                                  self.decay_batch_size))
        total_step = len(self.data_loader)

        perceptual_criterion = PerceptualLoss().to(self.device)
        content_criterion = nn.MSELoss().to(self.device)

        self.Encoder.train()
        self.Decoder.train()
        content_losses = AverageMeter()
        generator_losses = AverageMeter()
        perceptual_losses = AverageMeter()
        discriminator_losses = AverageMeter()
        ae_losses = AverageMeter()

        loss_window = self.visdom.line(Y=[0])
        gan_loss_window = self.visdom.line(Y=[0])
        content_loss_window = self.visdom.line(Y=[0])

        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        for epoch in range(self.epoch, self.num_epoch):
            content_losses.reset()
            perceptual_losses.reset()
            generator_losses.reset()
            ae_losses.reset()
            discriminator_losses.reset()
            for step, images in enumerate(self.data_loader):
                images = images.to(self.device)

                encoded_image = self.Encoder(images)

                binary_decoded_image = paq.compress(encoded_image.byte())
                encoded_image = paq.decompress(binary_decoded_image)

                decoded_image = self.Decoder(encoded_image)

                content_loss = content_criterion(images, decoded_image)
                perceptual_loss = perceptual_criterion(images, decoded_image)
                gan_loss = -self.Disciminator(decoded_image)

                ae_loss = content_loss * self.content_loss_factor + perceptual_loss * self.perceptual_loss_factor + \
                          gan_loss * self.generator_loss_factor

                content_losses.update(content_loss.item())
                perceptual_losses.update(perceptual_loss.item())
                generator_losses.update(gan_loss.item())
                ae_losses.update(ae_loss.item())

                optimizer_ae.zero_grad()
                ae_loss.backward()
                optimizer_ae.step()

                interpolated_image = self.eta * images + (1 - self.eta) * decoded_image
                gravity_penalty = self.Disciminator(interpolated_image)
                discriminator_loss = self.Disciminator(decoded_image) - self.Disciminator(images) +\
                                     gravity_penalty * self.penalty_loss_factor

                optimizer_discriminator.zero_grad()
                discriminator_loss.backward()
                optimizer_discriminator.step()

                discriminator_losses.update(discriminator_loss.item())

                if step % 100 == 0:
                    print(f"[Epoch {epoch}/{self.num_epoch}] [Batch {step}/{total_step}] "
                          f"[Content {content_loss:.4f}] [Perceptual] {perceptual_loss:.4f} [Gan {gan_loss}]"
                          f"[Discriminator {discriminator_loss}]")

                    save_image(images.cat(decoded_image, 2), f"Sample-epoch-{epoch}-step-{step}")

            loss_window = self.visdom.line(Y=ae_losses.avg, X=epoch, win=loss_window, update='append')
            gan_loss_window = self.visdom.line(Y=[generator_losses.avg, discriminator_losses.avg], X=epoch,
                                               win=gan_loss_window, update='append')
            content_loss_window = self.visdom.line(Y=content_losses.avg, X=epoch, win=content_loss_window,
                                                   update='append')
            lr_scheduler.step()

    def build_model(self):
        self.Encoder = Encoder(self.in_channels, self.storing_channels, self.nf)
        self.Decoder = Decoder(self.storing_channels, self.in_channels, self.nf)
        self.Disciminator = Discriminator()
        self.load_model()

    def load_model(self):
        print(f"[*] Load Model in {self.checkpoint_dir}")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        encoder_parameter_names = glob(os.path.join(self.checkpoint_dir, f"Encoder-{self.epoch - 1}.pth"))
        decoder_parameter_names = glob(os.path.join(self.checkpoint_dir, f"Decoder-{self.epoch - 1}.pth"))
        discriminator_parameters_names = glob(os.path.join(self.checkpoint_dir, f"Discriminator-{self.epoch - 1}.pth"))

        if not encoder_parameter_names or not decoder_parameter_names or not discriminator_parameters_names:
            print(f"[!] There is no parameter in {self.checkpoint_dir}")

        self.Encoder.load_state_dict(torch.load(encoder_parameter_names[0]))
        self.Decoder.load_state_dict(torch.load(decoder_parameter_names[0]))
        self.Disciminator.load_state_dict(torch.load(discriminator_parameters_names[0]))
