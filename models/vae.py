import math

import torch


class ConvVAE(torch.nn.Module):

    def __init__(self, in_size, filters, kernel_sizes, strides, paddings, latent_size, lamb=1e-2, bn=False):
        super().__init__()
        assert len(filters) == 2 and len(kernel_sizes) == 2 and len(strides) == 2 and len(paddings) == 2
        assert len(filters[0]) == len(kernel_sizes[0]) == len(strides[0]) == len(paddings[0])
        assert len(filters[1]) + 1 == len(kernel_sizes[1]) == len(strides[1]) == len(paddings[1])

        self.lamb = lamb
        self.bn = bn
        self.encoder_layers = torch.nn.ModuleList()
        if self.bn:
            self.encoder_bns = torch.nn.ModuleList()
        self.decoder_layers = torch.nn.ModuleList()
        if self.bn:
            self.decoder_bns = torch.nn.ModuleList()
        # encoder layers
        c_in = in_size[0]
        self.last_activation = torch.relu if c_in > 3 else torch.tanh
        h, w = in_size[1:]
        for num_filters, kernel_size, stride, padding in zip(filters[0], kernel_sizes[0], strides[0], paddings[0]):
            self.encoder_layers.append(torch.nn.Conv2d(c_in, num_filters, kernel_size, stride, padding))
            if self.bn:
                self.encoder_bns.append(torch.nn.BatchNorm2d(num_filters))
            c_in = num_filters
            h = math.floor((h + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
            w = math.floor((w + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
        self.h = h
        self.w = w
        self.c = c_in
        # encoder heads
        self.pre_latent_size = (self.c, self.h, self.w)
        self.mu = torch.nn.Linear(self.c * self.h * self.w, latent_size)
        self.logvar = torch.nn.Linear(self.c * self.h * self.w, latent_size)
        # decoder layers - assume the decoder is symmetrical to the encoder
        self.decoder_linear = torch.nn.Linear(latent_size, self.c * self.h * self.w)
        deconv_filters = filters[1] + [in_size[0]]
        for num_filters, kernel_size, stride, padding in zip(deconv_filters, kernel_sizes[1], strides[1], paddings[1]):
            self.decoder_layers.append(torch.nn.ConvTranspose2d(c_in, num_filters, kernel_size, stride, padding))
            if self.bn:
                self.decoder_bns.append(torch.nn.BatchNorm2d(num_filters) if bn else lambda x: x)
            c_in = num_filters
            h = (h - 1) * stride - 2 * padding + (kernel_size - 1) + 1
            w = (w - 1) * stride - 2 * padding + (kernel_size - 1) + 1
        if self.bn:
            del self.decoder_bns[-1]
        # TODO generalize this if necessary
        assert (h, w) == in_size[1:], \
            'set kernel size, padding and stride so that output and input sizes are the same; ' \
            f'in: {in_size[1:]} out: {h, w}'

    def encode(self, x):
        if self.bn:
            for layer, bn in zip(self.encoder_layers, self.encoder_bns):
                x = bn(torch.relu(layer(x)))
        else:
            for layer in self.encoder_layers:
                x = torch.relu(layer(x))
        x = x.view(-1, self.c * self.h * self.w)
        return self.mu(x), self.logvar(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        x = torch.relu(self.decoder_linear(z))
        x = x.view(-1, self.c, self.h, self.w)
        if self.bn:
            for layer, bn in zip(self.decoder_layers[:-1], self.decoder_bns):
                x = bn(torch.relu(layer(x)))
        else:
            for layer in self.decoder_layers[:-1]:
                x = torch.relu(layer(x))
        return self.last_activation(self.decoder_layers[-1](x))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reco_x = self.decode(z)
        return reco_x, mu, logvar, z

    def calculate_loss(self, X, X_recon, criterion, mu, logvar):
        rec_loss = criterion(X_recon, X)
        div_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        return rec_loss.mean() + self.lamb * div_loss, rec_loss, div_loss
