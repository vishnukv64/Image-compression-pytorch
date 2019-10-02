import os
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torchvision.utils import save_image
from glob import glob

from model.encoder import Encoder
from model.decoder import Decoder
from model.discriminator import Discriminator
from utils.utils import LambdaLR
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
        self.content_factor = config.content_factor
        self.perceptual_factor = config.perceptual_factor
        self.gan_loss_factor = config.gan_loss_factor
        self.discriminator_loss_factor = config.discriminator_loss_factor
        self.penalty_loss_factor = config.penalty_loss_factor

        self.checkpoint_dir = config.checkpoint_dir
        self.sample_dir = config.sample_dir
        self.epoch = config.epoch
        self.num_epoch = config.num_epoch

        self.data_loader = data_loader

    def train(self):
        optimizer_VAE = Adam([self.Encoder.parameters(), self.Decoder.parameters()], self.lr, betas=(self.b1, self.b2),
                             weight_decay=self.weight_decay)
        optimizer_discriminator = Adam(self.Disciminator.parameters(), self.lr, betas=(self.b1, self.b2),
                                       weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_VAE,
                                                         LambdaLR(self.num_epoch, self.epoch, len(self.data_loader),
                                                                  self.decay_batch_size))
        total_step = len(self.data_loader)

        perceptual_criterion = PerceptualLoss().to(self.device)
        content_criterion = nn.MSELoss().to(self.device)

        self.Encoder.train()
        self.Decoder.train()

        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        for epoch in range(self.epoch, self.num_epoch):
            for step, images in enumerate(self.data_loader):
                images = images.to(self.device)

                encoded_image = self.Encoder(images)

                decoded_image = self.Decoder(encoded_image)

                content_loss = content_criterion(images, decoded_image)
                perceptual_loss = perceptual_criterion(images, decoded_image)
                gan_loss = -self.Disciminator(decoded_image)

                VAE_loss = content_loss * self.content_factor + perceptual_loss * self.perceptual_factor + \
                           gan_loss * self.gan_loss_factor

                optimizer_VAE.zero_grad()
                VAE_loss.backward()
                optimizer_VAE.step()

                discriminator_loss = self.Disciminator(decoded_image) - self.Disciminator(images)
                optimizer_discriminator.zero_grad()
                discriminator_loss.backward()
                optimizer_discriminator.step()

                if step % 100 == 0:
                    print(f"[Epoch {epoch}/{self.num_epoch}] [Batch {step}/{total_step}] "
                          f"[Content {content_loss:.4f}] [Perceptual] {perceptual_loss:.4f} [Gan {gan_loss}]"
                          f"[Discriminator {discriminator_loss}]")

                    save_image()

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
