import torch.nn as nn
from torchvision.models.vgg import vgg16


class PerceptualLoss(nn.Module):
    def __init__(self, image_size=224):
        super(PerceptualLoss, self).__init__()

        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()
        self.up_sampler = nn.Upsample(size=image_size)

    def forward(self, real_image, decoded_image):
        high_resolution = self.up_sampler(real_image)
        fake_high_resolution = self.up_sampler(decoded_image)
        perceptual_loss = self.l1_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
        return perceptual_loss


class GradientPenalty(nn.Module):
    def __init__(self, eta, image_size=224):
        super(GradientPenalty, self).__init__()
        self.eta = eta

    def forward(self, real_image, fake_image):
        image = self.eta * real_image + (1 - self.eta) * fake_image

        return x