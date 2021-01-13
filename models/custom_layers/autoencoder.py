from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))

    # def encoder_no_update(self):
    #     for p in self.encoder.parameters():
    #         p.requires_grad = False


class AutoEncoder_adaptive(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder_adaptive, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, alpha):
        return self.decoder(self.encoder(x), alpha)

    # def encoder_no_update(self):
    #     for p in self.encoder.parameters():
    #         p.requires_grad = False
