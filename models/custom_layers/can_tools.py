import torch
from torch import nn
import torch.nn.functional as F

from itertools import product


def suppress_small_activation(x, phi, jump):
    # x.shape: batchsize,nb_atoms,L,L
    l1_norms = torch.sum(torch.abs(phi), dim=0)

    # to avoid division by zero
    l1_norms.add_(1e-12)

    # divide activations by L1 norms
    result = x / l1_norms.view(1, -1, 1, 1)

    result[(result < jump) * (result > -jump)] = 0.0

    result = result * l1_norms.view(1, -1, 1, 1)

    return result


class Saturation_activation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, jump):

        result = 0.5 * (torch.sign(x - jump) + torch.sign(x + jump))

        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Saturation_activation_per_filter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, phi, jump):
        # x.shape: batchsize,nb_atoms,L,L
        l1_norms = torch.sum(torch.abs(phi), dim=0)

        # to avoid division by zero
        l1_norms.add_(1e-12)

        # divide activations by L1 norms
        result = x / l1_norms.view(1, -1, 1, 1)

        result = 0.5 * (torch.sign(result - jump) + torch.sign(result + jump))

        result = result * l1_norms.view(1, -1, 1, 1)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class Take_max_band(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, phi, jump):
        # x.shape: batchsize,nb_atoms,L,L
        max_per_filter, max_per_filter_indices = x.abs().max(axis=1, keepdim=True)
        # max_per_filter.shape: batchsize,1,L,L

        l1_norms = torch.sum(torch.abs(phi), dim=0)

        # to avoid division by zero
        l1_norms.add_(1e-12)
        # l1_norms.shape=nb_atoms

        # divide activations by L1 norms
        result = x / l1_norms.view(1, -1, 1, 1)  # shape:1,nb_atom,1,1

        max_per_filter = max_per_filter / l1_norms[
            max_per_filter_indices.flatten()
            ].reshape(max_per_filter.shape)

        max_per_filter = max_per_filter.repeat_interleave(x.shape[1], axis=1)

        result[
            (result.abs() < (max_per_filter - jump))
            + (result.abs() > (max_per_filter + jump))
            ] = 0.0

        result = result * l1_norms.view(1, -1, 1, 1)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class Complicated_quantization_per_filter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, phi, jump):
        # x.shape: batchsize,nb_atoms,L,L
        max_per_filter, max_per_filter_indices = x.abs().max(axis=1, keepdim=True)
        # max_per_filter.shape: batchsize,1,L,L

        l1_norms = torch.sum(torch.abs(phi), dim=0)

        # to avoid division by zero
        l1_norms.add_(1e-12)
        # l1_norms.shape=nb_atoms

        # divide activations by L1 norms
        result = x / l1_norms.view(1, -1, 1, 1)  # shape:1,nb_atom,1,1

        result[(result < jump) * (result > -jump)] = 0.0

        max_per_filter = max_per_filter / l1_norms[
            max_per_filter_indices.flatten()
            ].reshape(max_per_filter.shape)

        max_per_filter = max_per_filter.repeat_interleave(x.shape[1], axis=1)

        result[
            (result.abs() < (max_per_filter - jump))
            + (result.abs() > (max_per_filter + jump))
            ] = 0.0

        result = 0.5 * (torch.sign(result - jump) + torch.sign(result + jump))

        result = result * l1_norms.view(1, -1, 1, 1)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def take_top_k(x, k):
    values, indices = torch.topk(x.abs(), k, dim=1)

    a, b, c = torch.meshgrid(
        torch.arange(x.shape[0], device=x.device),
        torch.arange(x.shape[2], device=x.device),
        torch.arange(x.shape[3], device=x.device),
        )
    a = torch.repeat_interleave(torch.unsqueeze(a, 1), k, axis=1)
    b = torch.repeat_interleave(torch.unsqueeze(b, 1), k, axis=1)
    c = torch.repeat_interleave(torch.unsqueeze(c, 1), k, axis=1)

    mask_indices = torch.stack(
        (a.flatten(), indices.flatten(), b.flatten(), c.flatten())
        )  # shape:(4,batchsize*L*L)

    mask = torch.sparse.FloatTensor(
        mask_indices, torch.ones(mask_indices.shape[1], device=x.device), x.shape
        ).to_dense()
    mask = mask.type(x.dtype)

    x = x * mask

    return x


class masking_BPDA_identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        return x * mask

    @staticmethod
    def backward(ctx, grad_wrt_output):
        return grad_wrt_output, None


class dropout_BPDA_identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dropout):
        return dropout(x)

    @staticmethod
    def backward(ctx, grad_wrt_output):
        return grad_wrt_output, None


def take_top_k_dropout(x, k, dropout, seed=None):
    values, indices = torch.topk(x.abs(), k, dim=1)

    a, b, c = torch.meshgrid(
        torch.arange(x.shape[0], device=x.device),
        torch.arange(x.shape[2], device=x.device),
        torch.arange(x.shape[3], device=x.device),
        )
    a = torch.repeat_interleave(torch.unsqueeze(a, 1), k, axis=1)
    b = torch.repeat_interleave(torch.unsqueeze(b, 1), k, axis=1)
    c = torch.repeat_interleave(torch.unsqueeze(c, 1), k, axis=1)

    mask_indices = torch.stack(
        (a.flatten(), indices.flatten(), b.flatten(), c.flatten())
        )  # shape:(4,batchsize*L*L)

    if seed:
        torch.manual_seed(seed)

    mask = torch.sparse.FloatTensor(
        mask_indices,
        dropout(torch.ones(mask_indices.shape[1], device=x.device)),
        x.shape,
        ).to_dense()
    mask = mask.type(x.dtype)

    if dropout.training:
        mask *= dropout.p

    # asd = masking_BPDA_identity().apply
    x = x * mask
    # x = asd(x, mask)

    return x


def take_top_coeff(x):  # across channels
    values, indices = torch.max(x.abs(), 1, keepdim=True)
    a, b, c = torch.meshgrid(
        torch.arange(x.shape[0], device=x.device),
        torch.arange(x.shape[2], device=x.device),
        torch.arange(x.shape[3], device=x.device),
        )

    mask_indices = torch.stack(
        (a.flatten(), indices.flatten(), b.flatten(), c.flatten())
        )  # shape:(4,batchsize*L*L)

    mask = torch.sparse.FloatTensor(
        mask_indices, torch.ones(mask_indices.shape[1], device=x.device), x.shape
        ).to_dense()
    mask = mask.type(x.dtype)

    x = x * mask

    return x


class take_top_coeff_BPDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):

        return take_top_coeff(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def s_to_patch(x, phi):
    # x.shape:batchsize,nb_atoms,L,L
    # phi.shape:patchsize*patchsize*3,nb_atoms
    out = torch.einsum("abcd,eb->acde", x, phi)
    # out.shape:batchsize,L,L,patchsize*patchsize*3

    # ########## BURASI GELEN RESIMLERI GORMEK ICIN
    # out = out.reshape(64, 15, 15, 4, 4, 3)
    # out = out.permute(0, 5, 1, 3, 2, 4)
    # out = out.reshape(64, 3, 60, 60)
    # out = out.permute(0, 2, 3, 1)
    # out = out.clamp(0, 1).cpu().numpy()

    # import matplotlib.pyplot as plt
    # import numpy as np

    # plt.figure(figsize=(10, 5))
    # indexing = np.arange(60)
    # indexing = indexing[np.tile([True, True, False, False], 15)]
    # subsampled = out[0, indexing]
    # subsampled = subsampled[:, indexing]
    # plt.imshow(subsampled)
    # plt.savefig("/home/canbakiskan/asd_s.pdf")
    # breakpoint()
    # ########## BURASI GELEN RESIMLERI GORMEK ICIN

    out = out.permute(0, 3, 1, 2)
    # out.shape:batchsize,patchsize*patchsize*3,L,L
    return out


def take_middle_of_img(x):
    width = x.shape[-1]
    start_index = (width - 32) // 2
    return x[:, :, start_index: start_index + 32, start_index: start_index + 32]


class decoder_base_class(nn.Module):
    def __init__(self, phi, jump):
        super(decoder_base_class, self).__init__()
        self.set_phi(phi)
        self.set_jump(jump)

    def set_jump(self, jump):
        if jump is not None:
            if isinstance(jump, torch.Tensor):
                self.jump = nn.Parameter(jump.float())
            else:
                self.jump = nn.Parameter(torch.tensor(jump, dtype=torch.float))
            self.jump.requires_grad = False

    def set_phi(self, phi):
        if phi is not None:
            if isinstance(phi, torch.Tensor):
                self.phi = nn.Parameter(phi.float())
            else:
                self.phi = nn.Parameter(torch.tensor(phi, dtype=torch.float))
            self.phi.requires_grad = False


class small_decoder(nn.Module):
    def __init__(self, n_input_channel, **kwargs):
        super(small_decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            n_input_channel, 100, kernel_size=4, stride=2, padding=0, bias=True
            )

        self.conv2 = nn.ConvTranspose2d(
            100, 3, kernel_size=3, stride=1, padding=0, bias=True
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = take_middle_of_img(out)
        return out


class large_decoder(nn.Module):
    def __init__(self, n_input_channel, **kwargs):

        super(large_decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            n_input_channel, 300, kernel_size=4, stride=2, padding=0, bias=True
            )
        self.conv2 = nn.ConvTranspose2d(
            300, 100, kernel_size=3, stride=1, padding=0, bias=True
            )
        self.conv3 = nn.ConvTranspose2d(
            100, 3, kernel_size=3, stride=1, padding=0, bias=True
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = take_middle_of_img(out)
        return out


class resize_decoder(nn.Module):
    def __init__(self, n_input_channel, **kwargs):

        super(resize_decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            n_input_channel, 300, kernel_size=3, stride=1, padding=0, bias=True
            )

        self.conv2 = nn.ConvTranspose2d(
            300, 100, kernel_size=3, stride=1, padding=0, bias=True
            )
        self.conv3 = nn.ConvTranspose2d(
            100, 3, kernel_size=3, stride=1, padding=0, bias=True
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.interpolate(out, size=34, mode="bicubic")
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = take_middle_of_img(out)
        return out


class top_coeff_decoder(decoder_base_class):
    """
    Takes the top coefficient among all atom coefficients, 
    then feeds into large decoder
    """

    def __init__(self, n_input_channel, **kwargs):
        super(top_coeff_decoder, self).__init__(None, None)
        self.decoder = large_decoder(n_input_channel)

    def forward(self, x):
        out = take_top_coeff(x)
        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        return self.decoder(out)


class top_coeff_quant_decoder(decoder_base_class):
    """
    Takes the top coefficient among all atom coefficients quantizes it, 
    then feeds into large decoder
    """

    def __init__(self, n_input_channel, phi, jump, **kwargs):
        super(top_coeff_quant_decoder, self).__init__(phi, jump)
        self.activation = Saturation_activation_per_filter().apply
        self.decoder = large_decoder(n_input_channel)

    def forward(self, x):
        out = take_top_coeff(x)
        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = self.activation(out, self.phi, self.jump)
        return self.decoder(out)


class reconstruction_decoder(decoder_base_class):
    """
    Takes !reconstructions! and feeds into large decoder
    """

    def __init__(self, **kwargs):
        super(reconstruction_decoder, self).__init__(None, None)
        self.decoder = large_decoder(3)

    def forward(self, x):
        return self.decoder(x)


class top_patch_decoder(decoder_base_class):
    """
    Takes the top coefficient among all atom coefficients then 
    multiplies with dictionary phi, and then feeds into large decoder
    """

    def __init__(self, n_input_channel, phi, **kwargs):

        super(top_patch_decoder, self).__init__(phi, None)
        self.decoder = large_decoder(n_input_channel)

    def forward(self, x):
        out = take_top_coeff(x)
        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = s_to_patch(out, self.phi)
        # out.shape: batchsize,patchsize*patchsize*3,L,L

        return self.decoder(out)


class top10_quant_patch_avg_decoder(decoder_base_class):
    def __init__(self, n_input_channel, phi, jump, **kwargs):

        super(top10_quant_patch_avg_decoder, self).__init__(phi, jump)
        self.decoder = large_decoder(n_input_channel)
        self.activation = Saturation_activation_per_filter().apply

    def forward(self, x):
        out = take_top_k(x, 10)
        out = self.activation(out, self.phi, self.jump)

        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = s_to_patch(out, self.phi) / 10
        # out.shape: batchsize,patchsize*patchsize*3,L,L

        return self.decoder(out)


class top_patch_quant_decoder(decoder_base_class):
    """
    Takes the top coefficient among all atom coefficients then quantizes it,
    then multiplies with dictionary phi, and feeds into large decoder
    """

    def __init__(self, n_input_channel, phi, jump, **kwargs):

        super(top_patch_quant_decoder, self).__init__(phi, jump)
        self.activation = Saturation_activation_per_filter().apply

        self.decoder = large_decoder(n_input_channel)

    def forward(self, x):
        out = take_top_coeff(x)
        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = self.activation(out, self.phi, self.jump)
        out = s_to_patch(out, self.phi)
        # out.shape: batchsize,patchsize*patchsize*3,L,L

        return self.decoder(out)


class quant_decoder(decoder_base_class):
    """
    Quantizes coefficients then feeds into large decoder
    """

    def __init__(self, n_input_channel, phi, jump, **kwargs):
        super(quant_decoder, self).__init__(phi, jump)
        self.activation = Saturation_activation_per_filter().apply
        self.decoder = large_decoder(n_input_channel)

    def forward(self, x):
        out = self.activation(x, self.phi, self.jump)

        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = self.decoder(out)

        return out


class patch_quant_decoder(decoder_base_class):
    """
    Quantizes all coefficients, then multiplies with dictionary phi,
     and feeds into large decoder
    """

    def __init__(self, n_input_channel, phi, jump, **kwargs):

        super(patch_quant_decoder, self).__init__(phi, jump)
        self.activation = Saturation_activation_per_filter().apply
        self.decoder = large_decoder(n_input_channel)

    def forward(self, x):
        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = self.activation(x, self.phi, self.jump)
        out = s_to_patch(out, self.phi)
        # out.shape: batchsize,patchsize*patchsize*3,L,L

        return self.decoder(out)


class patch_decoder(decoder_base_class):
    """
    Multiplies coefficients with dictionary phi and feeds into large decoder
    """

    def __init__(self, n_input_channel, phi, jump, **kwargs):

        super(patch_decoder, self).__init__(phi, None)
        self.decoder = large_decoder(n_input_channel)

    def forward(self, x):
        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = s_to_patch(x, self.phi)
        # out.shape: batchsize,patchsize*patchsize*3,L,L

        return self.decoder(out)


class suppress_decoder(decoder_base_class):
    """
    Suppresses coefficients that are smaller than the threshold,
    then feeds into large decoder
    """

    def __init__(self, n_input_channel, phi, jump, **kwargs):
        super(suppress_decoder, self).__init__(phi, jump)
        self.decoder = large_decoder(n_input_channel)

    def forward(self, x):
        out = suppress_small_activation(x, self.phi, self.jump)

        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = self.decoder(out)

        return out


class top30_decoder(decoder_base_class):
    """
    Quantizes coefficients based on ||phi_i||_1
    """

    def __init__(self, n_input_channel, phi, jump, **kwargs):
        super(top30_decoder, self).__init__(phi, jump)
        self.activation = Saturation_activation_per_filter().apply
        self.decoder = large_decoder(n_input_channel)

    def forward(self, x):
        # out = self.activation(x, self.phi, self.jump)
        out = take_top_k(x, 30)
        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = self.decoder(out)

        return out


class top30_quant_decoder(decoder_base_class):
    """
    Quantizes coefficients based on ||phi_i||_1
    """

    def __init__(self, n_input_channel, phi, jump, **kwargs):
        super(top30_quant_decoder, self).__init__(phi, jump)
        self.activation = Saturation_activation_per_filter().apply
        self.decoder = large_decoder(n_input_channel)

    def forward(self, x):
        # out = self.activation(x, self.phi, self.jump)
        out = take_top_k(x, 30)
        out = self.activation(out, self.phi, self.jump)
        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = self.decoder(out)

        return out


class top30_dropout_quant_decoder(decoder_base_class):
    """
    """

    def __init__(self, n_input_channel, phi, jump, p, **kwargs):
        super(top30_dropout_quant_decoder, self).__init__(phi, jump)
        self.activation = Saturation_activation_per_filter().apply
        self.decoder = large_decoder(n_input_channel)
        self.dropout = nn.Dropout(p=p)
        self.seed = None

    def fix_seed(self, seed):
        self.seed = seed

    def rm_seed(self):
        self.seed = None

    def forward(self, x):
        # out = self.activation(x, self.phi, self.jump)
        out = take_top_k_dropout(x, 30, self.dropout, self.seed)
        out = self.activation(out, self.phi, self.jump)
        if self.training:
            out = out / self.dropout.p
        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = self.decoder(out)

        return out


class top10_dropout_quant_decoder(decoder_base_class):
    """
    """

    def __init__(self, n_input_channel, phi, jump, p, **kwargs):
        super(top10_dropout_quant_decoder, self).__init__(phi, jump)
        self.activation = Saturation_activation_per_filter().apply
        self.decoder = large_decoder(n_input_channel)
        self.dropout = nn.Dropout(p=p)
        self.seed = None

    def fix_seed(self, seed):
        self.seed = seed

    def rm_seed(self):
        self.seed = None

    def forward(self, x):
        # out = self.activation(x, self.phi, self.jump)
        out = take_top_k_dropout(x, 10, self.dropout, self.seed)
        out = self.activation(out, self.phi, self.jump)
        if self.training:
            out = out / self.dropout.p
        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = self.decoder(out)

        return out


class band_top10_dropout_quant_decoder(decoder_base_class):
    """
    """

    def __init__(self, n_input_channel, phi, jump, p, **kwargs):
        super(band_top10_dropout_quant_decoder, self).__init__(phi, jump)
        self.max_band = Take_max_band().apply
        self.activation = Saturation_activation_per_filter().apply
        self.decoder = large_decoder(n_input_channel)
        self.dropout = nn.Dropout(p=p)
        self.seed = None

    def fix_seed(self, seed):
        self.seed = seed

    def rm_seed(self):
        self.seed = None

    def forward(self, x):
        out = self.max_band(x, self.phi, self.jump)
        out = take_top_k_dropout(out, 10, self.dropout, self.seed)
        out = self.activation(out, self.phi, 3 * self.jump)
        if self.training:
            out = out / self.dropout.p
        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = self.decoder(out)

        return out


class band_top30_dropout_quant_decoder(decoder_base_class):
    """
    """

    def __init__(self, n_input_channel, phi, jump, p, **kwargs):
        super(band_top30_dropout_quant_decoder, self).__init__(phi, jump)
        self.max_band = Take_max_band().apply
        self.activation = Saturation_activation_per_filter().apply
        self.decoder = large_decoder(n_input_channel)
        self.dropout = nn.Dropout(p=p)
        self.seed = None

    def fix_seed(self, seed):
        self.seed = seed

    def rm_seed(self):
        self.seed = None

    def forward(self, x):
        out = self.max_band(x, self.phi, self.jump)
        out = take_top_k_dropout(out, 30, self.dropout, self.seed)
        out = self.activation(out, self.phi, 3 * self.jump)
        if self.training:
            out = out / self.dropout.p
        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = self.decoder(out)

        return out


class band_top50_dropout_quant_decoder(decoder_base_class):
    """
    """

    def __init__(self, n_input_channel, phi, jump, p, **kwargs):
        super(band_top50_dropout_quant_decoder, self).__init__(phi, jump)
        self.max_band = Take_max_band().apply
        self.activation = Saturation_activation_per_filter().apply
        self.decoder = large_decoder(n_input_channel)
        self.dropout = nn.Dropout(p=p)
        self.seed = None

    def fix_seed(self, seed):
        self.seed = seed

    def rm_seed(self):
        self.seed = None

    def forward(self, x):
        out = self.max_band(x, self.phi, self.jump)
        out = take_top_k_dropout(out, 50, self.dropout, self.seed)
        out = self.activation(out, self.phi, 3 * self.jump)
        if self.training:
            out = out / self.dropout.p
        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = self.decoder(out)

        return out


class band_top75_dropout_quant_decoder(decoder_base_class):
    """
    """

    def __init__(self, n_input_channel, phi, jump, p, **kwargs):
        super(band_top75_dropout_quant_decoder, self).__init__(phi, jump)
        self.max_band = Take_max_band().apply
        self.activation = Saturation_activation_per_filter().apply
        self.decoder = large_decoder(n_input_channel)
        self.dropout = nn.Dropout(p=p)
        self.seed = None

    def fix_seed(self, seed):
        self.seed = seed

    def rm_seed(self):
        self.seed = None

    def forward(self, x):
        out = self.max_band(x, self.phi, self.jump)
        out = take_top_k_dropout(x, 75, self.dropout, self.seed)
        out = self.activation(out, self.phi, 3 * self.jump)
        if self.training:
            out = out / self.dropout.p
        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = self.decoder(out)

        return out


class top100_quant_decoder(decoder_base_class):
    """
    Quantizes coefficients based on ||phi_i||_1
    """

    def __init__(self, n_input_channel, phi, jump, **kwargs):
        super(top100_quant_decoder, self).__init__(phi, jump)
        self.activation = Saturation_activation_per_filter().apply
        self.decoder = large_decoder(n_input_channel)

    def forward(self, x):
        # out = self.activation(x, self.phi, self.jump)
        out = take_top_k(x, 100)
        out = self.activation(out, self.phi, self.jump)
        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = self.decoder(out)

        return out


class top5_quant_decoder(decoder_base_class):
    """
    Quantizes coefficients based on ||phi_i||_1
    """

    def __init__(self, n_input_channel, phi, jump, **kwargs):
        super(top5_quant_decoder, self).__init__(phi, jump)
        self.activation = Saturation_activation_per_filter().apply
        self.decoder = large_decoder(n_input_channel)

    def forward(self, x):
        # out = self.activation(x, self.phi, self.jump)
        out = take_top_k(x, 5)
        out = self.activation(out, self.phi, self.jump)
        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = self.decoder(out)

        return out


class top1_quant_decoder(decoder_base_class):
    """
    Quantizes coefficients based on ||phi_i||_1
    """

    def __init__(self, n_input_channel, phi, jump, **kwargs):
        super(top1_quant_decoder, self).__init__(phi, jump)
        self.activation = Saturation_activation_per_filter().apply
        self.decoder = large_decoder(n_input_channel)

    def forward(self, x):
        # out = self.activation(x, self.phi, self.jump)
        out = take_top_k(x, 1)
        out = self.activation(out, self.phi, self.jump)
        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = self.decoder(out)

        return out


class top2_quant_decoder(decoder_base_class):
    """
    Quantizes coefficients based on ||phi_i||_1
    """

    def __init__(self, n_input_channel, phi, jump, **kwargs):
        super(top2_quant_decoder, self).__init__(phi, jump)
        self.activation = Saturation_activation_per_filter().apply
        self.decoder = large_decoder(n_input_channel)

    def forward(self, x):
        # out = self.activation(x, self.phi, self.jump)
        out = take_top_k(x, 2)
        out = self.activation(out, self.phi, self.jump)
        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = self.decoder(out)

        return out


class complicated_quant_decoder(decoder_base_class):
    """
    Quantizes coefficients based on ||phi_i||_1
    """

    def __init__(self, n_input_channel, phi, jump, **kwargs):
        super(complicated_quant_decoder, self).__init__(phi, jump)
        self.activation = Complicated_quantization_per_filter().apply
        self.decoder = large_decoder(n_input_channel)

    def forward(self, x):
        out = self.activation(x, self.phi, self.jump)
        # out shape same as x shape i.e. batchsize,nb_atoms,L,L
        out = self.decoder(out)

        return out


class null_decoder(nn.Module):
    def __init__(self, **kwargs):
        super(null_decoder, self).__init__()
        # self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x


# class top30_quant_patch_median_decoder(decoder_base_class):
#     def __init__(self, n_input_channel, phi, jump, **kwargs):

#         super(top30_quant_patch_median_decoder, self).__init__(phi, jump)
#         self.decoder = large_decoder(n_input_channel)
#         self.activation = Saturation_activation_per_filter().apply

#     def forward(self, x):
#         out = take_top_k(x, 30) # shape: batchsize, nbatom, L,L
#         # batchsize, nbatom, patchshape, L,L
#         # sort in nbatom dimension, take 15th for each pixel
#         torch.sort(input, dim=-1, descending=False, out=None)
#         out = self.activation(out, self.phi, self.jump)
#         out =

#         # out shape same as x shape i.e. batchsize,nb_atoms,L,L
#         out = s_to_patch(out, self.phi) / 10
#         # out.shape: batchsize,patchsize*patchsize*3,L,L

#         return self.decoder(out)


decoder_dict = {
    "small_decoder": small_decoder,
    "large_decoder": large_decoder,
    "resize_decoder": resize_decoder,
    "top_coeff_decoder": top_coeff_decoder,
    "top_coeff_quant_decoder": top_coeff_quant_decoder,
    "reconstruction_decoder": reconstruction_decoder,
    "top_patch_decoder": top_patch_decoder,
    "top_patch_quant_decoder": top_patch_quant_decoder,
    "quant_decoder": quant_decoder,
    "patch_quant_decoder": patch_quant_decoder,
    "patch_decoder": patch_decoder,
    "suppress_decoder": suppress_decoder,
    "top30_decoder": top30_decoder,
    "top30_dropout_quant_decoder": top30_dropout_quant_decoder,
    "top10_dropout_quant_decoder": top10_dropout_quant_decoder,
    "band_top50_dropout_quant_decoder": band_top50_dropout_quant_decoder,
    "band_top30_dropout_quant_decoder": band_top30_dropout_quant_decoder,
    "band_top10_dropout_quant_decoder": band_top10_dropout_quant_decoder,
    "band_top75_dropout_quant_decoder": band_top75_dropout_quant_decoder,
    "top30_quant_decoder": top30_quant_decoder,
    "top1_quant_decoder": top1_quant_decoder,
    "top2_quant_decoder": top2_quant_decoder,
    "top5_quant_decoder": top5_quant_decoder,
    "top100_quant_decoder": top100_quant_decoder,
    "complicated_quant_decoder": complicated_quant_decoder,
    "top10_quant_patch_avg_decoder": top10_quant_patch_avg_decoder,
    "null_decoder": null_decoder,
}
