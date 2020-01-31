import math

import torch


def squared_euclidean_distances(X, Y):
    m1 = X.unsqueeze(1).repeat(1, Y.size(0), 1)
    m2 = Y.unsqueeze(0).repeat(X.size(0), 1, 1)
    return (m1 - m2).pow(2).sum(2)


def mmd2(K_XX, K_YY, K_XY, biased=False):
    # for alternative implementation:
    # https://discuss.pytorch.org/t/maximum-mean-discrepancy-mmd-and-radial-basis-function-rbf/1875/2
    # for biased / unbiased see:
    # http://www.stat.cmu.edu/~ryantibs/journalclub/mmd.pdf
    m = K_XX.size(0)
    assert m == K_XX.size(1)
    n = K_YY.size(0)
    assert n == K_YY.size(1)
    assert (m == K_XY.size(0) and n == K_XY.size(1)) or (n == K_XY.size(0) and m == K_XY.size(1))
    if not biased:
        diag_X = torch.diag(K_XX)
        diag_Y = torch.diag(K_YY)
        K_XX_summed = K_XX.sum() - diag_X.sum()
        K_YY_summed = K_YY.sum() - diag_Y.sum()
        K_XY_summed = K_XY.sum()
        mmd2 = K_XX_summed / (m * (m - 1)) + K_YY_summed / (n * (n - 1)) - 2 * K_XY_summed / (n * m)
    else:
        K_XX_summed = K_XX.sum()
        K_YY_summed = K_YY.sum()
        K_XY_summed = K_XY.sum()
        mmd2 = (K_XX_summed / (m ** 2) + K_YY_summed / (n ** 2) - 2 * K_XY_summed / (n * m)) ** 0.5
    return mmd2


def imq_kernel(X, Y, scale_list=None, p_variance=None):
    # inverse multiquadratics kernel
    # k(x, y) = C / (C + ||x - y||^2)
    # https://arxiv.org/pdf/1711.01558.pdf
    # https://github.com/tolstikhin/wae/blob/master/wae.py
    # https://github.com/tolstikhin/wae/blob/master/improved_wae.py
    # https://github.com/1Konny/WAE-pytorch/blob/master/ops.py
    # scale_list = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0] if scale_list is None else scale_list
    scale_list = [1.0] if scale_list is None else scale_list
    z_dim = X.size(1)
    squared_distances = squared_euclidean_distances(X, Y)
    res = torch.zeros_like(squared_distances)
    if p_variance is not None:
        # assume normal p_z
        c_base = 2.0 * z_dim * p_variance
    else:
        c_base = z_dim
    for scale in scale_list:
        c = c_base * scale
        res.add_(c / (c + squared_distances))
    return res


class ConvWAE(torch.nn.Module):

    def __init__(self, in_size, filters, kernel_sizes, strides, paddings, latent_size, bn=False, lamb=10, z_var=2.0):
        super().__init__()
        assert len(filters) == 2 and len(kernel_sizes) == 2 and len(strides) == 2 and len(paddings) == 2
        assert len(filters[0]) == len(kernel_sizes[0]) == len(strides[0]) == len(paddings[0])
        assert len(filters[1]) + 1 == len(kernel_sizes[1]) == len(strides[1]) == len(paddings[1])

        self.lamb = lamb
        self.z_dim = latent_size
        self.z_var = z_var
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
        self.pre_latent_size = (self.c, self.h, self.w)
        self.encoder_layers.append(torch.nn.Linear(self.c * self.h * self.w, latent_size))
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
        assert (h, w) == in_size[1:], \
            'set kernel size, padding and stride so that output and input sizes are the same; ' \
            f'in: {in_size[1:]} out: {h, w}'

    def encode(self, x):
        if self.bn:
            for layer, bn in zip(self.encoder_layers[:-1], self.encoder_bns):
                x = bn(torch.relu(layer(x)))
        else:
            for layer in self.encoder_layers[:-1]:
                x = torch.relu(layer(x))
        x = x.view(-1, self.c * self.h * self.w)
        return self.encoder_layers[-1](x)

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
        z = self.encode(x)
        reco_x = self.decode(z)
        return reco_x, z

    def calculate_loss(self, X, X_recon, criterion, z_gen, z_enc, biased=False):
        rec_loss = criterion(X_recon, X)
        dis_loss = self._discrepancy_loss(z_gen, z_enc, biased=biased)
        return rec_loss.mean() + self.lamb * dis_loss, rec_loss, dis_loss

    def _discrepancy_loss(self, z_gen, z_enc, biased=False):
        K_XX = imq_kernel(z_gen, z_gen, p_variance=self.z_var)
        K_YY = imq_kernel(z_enc, z_enc, p_variance=self.z_var)
        K_XY = imq_kernel(z_gen, z_enc, p_variance=self.z_var)
        return mmd2(K_XX, K_YY, K_XY, biased=biased)
