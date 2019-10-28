import os
import paq
import torch
import torch.nn as nn
import numpy as np
from torch.optim.adam import Adam
from torchvision.utils import save_image
from glob import glob
from visdom import Visdom
from itertools import chain

from model.encoder import Encoder
from model.decoder import Decoder
from model.discriminator import Discriminator
from utils.utils import LambdaLR, AverageMeter, create_vis_plot, update_vis_plot, get_lr
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
        self.decay_epoch = config.decay_epoch
        self.content_loss_factor = config.content_loss_factor
        self.perceptual_loss_factor = config.perceptual_loss_factor
        self.generator_loss_factor = config.generator_loss_factor
        self.discriminator_loss_factor = config.discriminator_loss_factor
        self.penalty_loss_factor = config.penalty_loss_factor
        self.rate_loss_factor = config.rate_loss_factor
        self.eta = config.eta

        self.checkpoint_dir = config.checkpoint_dir
        self.sample_dir = config.sample_dir
        self.epoch = config.epoch
        self.num_epoch = config.num_epoch
        self.image_size = config.image_size

        self.data_loader = data_loader

        self.content_losses = AverageMeter()
        self.generator_losses = AverageMeter()
        self.perceptual_losses = AverageMeter()
        self.discriminator_losses = AverageMeter()
        self.ae_losses = AverageMeter()

        self.visdom = Visdom()

        self.build_model()

    def train(self):
        optimizer_ae = Adam(chain(self.Encoder.parameters(), self.Decoder.parameters()), self.lr,
                            betas=(self.b1, self.b2),
                            weight_decay=self.weight_decay)
        optimizer_discriminator = Adam(self.Disciminator.parameters(), self.lr, betas=(self.b1, self.b2),
                                       weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_ae,
                                                         LambdaLR(self.num_epoch, self.epoch, self.decay_epoch).step)
        total_step = len(self.data_loader)

        perceptual_criterion = PerceptualLoss().to(self.device)
        content_criterion = nn.L1Loss().to(self.device)
        adversarial_criterion = nn.BCELoss().to(self.device)
        rate_criterion = nn.BCELoss().to(self.device)

        self.Encoder.train()
        self.Decoder.train()
        content_losses = AverageMeter()
        generator_losses = AverageMeter()
        perceptual_losses = AverageMeter()
        discriminator_losses = AverageMeter()
        ae_losses = AverageMeter()
        rate_losses = AverageMeter()

        lr_window = create_vis_plot('Epoch', 'Learning rate', 'Learning rate')
        loss_window = create_vis_plot('Epoch', 'Loss', 'Total Loss')
        generator_loss_window = create_vis_plot('Epoch', 'Loss', 'Generator Loss')
        discriminator_loss_window = create_vis_plot('Epoch', 'Loss', 'Discriminator Loss')
        content_loss_window = create_vis_plot('Epoch', 'Loss', 'Content Loss')
        perceptual_loss_window = create_vis_plot('Epoch', 'Loss', 'Perceptual Loss')
        rate_loss_window = create_vis_plot('Epoch', 'Loss', 'Rate Loss')

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
            rate_losses.reset()
            for step, images in enumerate(self.data_loader):
                images = images.to(self.device)

                real_labels = torch.ones((images.size(0), 1)).to(self.device)
                fake_labels = torch.zeros((images.size(0), 1)).to(self.device)

                encoded_image = self.Encoder(images)

                binary_decoded_image = paq.compress(encoded_image.cpu().detach().numpy().tobytes())
                # encoded_image = paq.decompress(binary_decoded_image)
                #
                # encoded_image = torch.from_numpy(np.frombuffer(encoded_image, dtype=np.float32)
                #                                  .reshape(-1, self.storing_channels, self.image_size // 8,
                #                                           self.image_size // 8)).to(self.device)

                decoded_image = self.Decoder(encoded_image)

                content_loss = content_criterion(images, decoded_image)
                perceptual_loss = perceptual_criterion(images, decoded_image)
                generator_loss = adversarial_criterion(self.Disciminator(decoded_image), real_labels)
                # generator_loss = -self.Disciminator(decoded_image).mean()

                ae_loss = content_loss * self.content_loss_factor + perceptual_loss * self.perceptual_loss_factor + \
                          generator_loss * self.generator_loss_factor + rate_criterion * self.rate_loss_factor

                content_losses.update(content_loss.item())
                perceptual_losses.update(perceptual_loss.item())
                generator_losses.update(generator_loss.item())
                ae_losses.update(ae_loss.item())
                rate_losses.update(real_loss.item())

                optimizer_ae.zero_grad()
                ae_loss.backward(retain_graph=True)
                optimizer_ae.step()

                interpolated_image = self.eta * images + (1 - self.eta) * decoded_image
                gravity_penalty = self.Disciminator(interpolated_image).mean()
                real_loss = adversarial_criterion(self.Disciminator(images), real_labels)
                fake_loss = adversarial_criterion(self.Disciminator(decoded_image), fake_labels)
                discriminator_loss = (real_loss + fake_loss) * self.discriminator_loss_factor / 2 +\
                                     gravity_penalty * self.penalty_loss_factor

                # discriminator_loss = self.Disciminator(decoded_image).mean() - self.Disciminator(images).mean() + \
                #                      gravity_penalty * self.penalty_loss_factor

                optimizer_discriminator.zero_grad()
                discriminator_loss.backward(retain_graph=True)
                optimizer_discriminator.step()
                discriminator_losses.update(discriminator_loss.item())

                if step % 100 == 0:
                    print(f"[Epoch {epoch}/{self.num_epoch}] [Batch {step}/{total_step}] [Learning rate {get_lr(optimizer_ae)}] "
                          f"[Content {content_loss:.4f}] [Perceptual {perceptual_loss:.4f}] [Gan {generator_loss:.4f}]"
                          f"[Discriminator {discriminator_loss:.4f}]")

                    save_image(torch.cat([images, decoded_image], dim=2),
                               os.path.join(self.sample_dir, f"Sample-epoch-{epoch}-step-{step}.png"))

            update_vis_plot(epoch, ae_losses.avg, loss_window, 'append')
            update_vis_plot(epoch, generator_losses.avg, generator_loss_window, 'append')
            update_vis_plot(epoch, discriminator_losses.avg, discriminator_loss_window, 'append')
            update_vis_plot(epoch, content_losses.avg, content_loss_window, 'append')
            update_vis_plot(epoch, perceptual_losses.avg, perceptual_loss_window, 'append')
            update_vis_plot(epoch, get_lr(optimizer_ae), lr_window, 'append')
            update_vis_plot(epoch, rate_losses.avg, rate_loss_window, 'append')

            lr_scheduler.step()

            torch.save(self.Encoder.state_dict(), os.path.join(self.checkpoint_dir, f"Encoder-{epoch}.pth"))
            torch.save(self.Decoder.state_dict(), os.path.join(self.checkpoint_dir, f"Decoder-{epoch}.pth"))
            torch.save(self.Disciminator.state_dict(), os.path.join(self.checkpoint_dir, f"Discriminator-{epoch}.pth"))

    def build_model(self):
        self.Encoder = Encoder(self.in_channels, self.storing_channels, self.nf).to(self.device)
        self.Decoder = Decoder(self.storing_channels, self.in_channels, self.nf).to(self.device)
        self.Disciminator = Discriminator().to(self.device)
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
            return

        self.Encoder.load_state_dict(torch.load(encoder_parameter_names[0]))
        self.Decoder.load_state_dict(torch.load(decoder_parameter_names[0]))
        self.Disciminator.load_state_dict(torch.load(discriminator_parameters_names[0]))
