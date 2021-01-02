import numpy as np

import torch
from torch.nn import functional as F
from torch.nn import Parameter
from torch.nn.modules import Conv2d, Module


class GaussianConv2d(Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
            sigma=1.0,
            ):
        super().__init__()

        self.sigma = sigma
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = torch.zeros((out_channels, in_channels//groups,
                                   self.kernel_size[0], self.kernel_size[1]))

        self.initialize_kernels()
        if bias:
            self.initialize_bias()
        else:
            self.bias = None

        self.register_parameter("weight", self.weight)
        # self.register_parameter("bias", self.bias)

    def forward(self, input_tensor):
        return F.conv2d(input_tensor, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def initialize_kernels(self):
        gauss = GaussianKernel2d(self.kernel_size, self.sigma)
        self.weight = gauss.generate_torch_kernel()
        self.weight = Parameter(self.weight, requires_grad=False)

    def initialize_bias(self):
        self.bias = Parameter(torch.zeros((self.out_channels)), requires_grad=True)

    def normalize_kernels(self):
        self.weight = Parameter(self.weight /
                                torch.norm(self.weight.view(self.weight.size(0), -1),
                                           p=1, dim=1, keepdim=True).unsqueeze(1).unsqueeze(1), requires_grad=False)


class DifferenceOfGaussianConv2d(Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
            sigma1=0.5,
            sigma2=None
            ):
        super().__init__()

        self.sigma1 = sigma1
        if sigma2:
            self.sigma2 = sigma2
        else:
            self.sigma2 = sigma1*1.6
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = torch.zeros((out_channels, in_channels//groups,
                                   self.kernel_size[0], self.kernel_size[1]))

        self.initialize_kernels()
        if bias:
            self.initialize_bias()
        else:
            self.bias = None

        self.register_parameter("weight", self.weight)
        # self.register_parameter("bias", self.bias)

    def forward(self, input_tensor):
        return F.conv2d(input_tensor, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def initialize_kernels(self):
        dog = DifferenceOfGaussian2d(self.kernel_size, self.sigma1, self.sigma2)
        self.weight = dog.generate_torch_kernel()
        self.weight = Parameter(self.weight, requires_grad=False)

    def initialize_bias(self):
        self.bias = Parameter(torch.zeros((self.out_channels)), requires_grad=True)

    def normalize_kernels(self):
        self.weight = Parameter(self.weight /
                                torch.norm(self.weight.view(self.weight.size(0), -1),
                                           p=1, dim=1, keepdim=True).unsqueeze(1).unsqueeze(1), requires_grad=False)


"""
Gaussian Filter
Difference of Gaussian Filter
"""


class GaussianKernel2d(object):
    """docstring for GaussianKernel2d"""

    def __init__(self, kernel_size, sigma, normalize=True):
        super(GaussianKernel2d, self).__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.sigma = sigma
        self.normalize = normalize

    @property
    def kernel(self):
        return self.generate_kernel()

    def generate_kernel(self):

        x_axis = np.arange(-(self.kernel_size[0]//2), self.kernel_size[0]//2+1)
        y_axis = np.arange(-(self.kernel_size[1]//2), self.kernel_size[1]//2+1)
        x, y = np.meshgrid(x_axis, y_axis)
        kernel = np.zeros((self.kernel_size[0], self.kernel_size[0]))
        for i, xi in enumerate(x_axis):
            for j, yi in enumerate(y_axis):
                kernel[i, j] = self.gaussian(np.array([xi, yi]), np.array(
                    [0, 0]), np.array([[self.sigma**2, 0.], [0., self.sigma**2]]))
        if self.normalize:
            kernel /= np.sum(kernel)
        return kernel

    def generate_torch_kernel(self):
        return torch.Tensor(self.generate_kernel())

    @staticmethod
    def gaussian(x, mu, sigma):

        if isinstance(x, int) or isinstance(x, float):
            x = np.array([x])
        if isinstance(mu, int) or isinstance(mu, float):
            mu = np.array([mu])
        if isinstance(sigma, int) or isinstance(sigma, float):
            sigma = np.array([[sigma]])

        dimension = x.shape[0]
        normalization_term = 1/np.sqrt(((2*np.pi)**dimension) * np.linalg.det(sigma))
        exponential_term = np.exp(-1./2 * (x-mu).T @ np.linalg.inv(sigma) @ (x-mu))

        return normalization_term * exponential_term

    def plot(self):

        from matplotlib import pyplot as plt
        from matplotlib import cm

        x_axis = np.arange(-(self.kernel_size//2), self.kernel_size//2+1)
        y_axis = np.arange(-(self.kernel_size//2), self.kernel_size//2+1)
        x, y = np.meshgrid(x_axis, y_axis)

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        surf = ax.plot_surface(x, y, self.kernel, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        plt.show()


class DifferenceOfGaussian2d(GaussianKernel2d):
    """docstring for DifferenceOfGaussian2d"""

    def __init__(self, kernel_size, sigma1, sigma2=None, normalize=True):
        super().__init__(kernel_size, sigma1, normalize)
        self.sigma1 = sigma1
        if sigma2:
            self.sigma2 = sigma2
        else:
            self.sigma2 = sigma1*1.6

        self.G1 = GaussianKernel2d(kernel_size, self.sigma1, False)
        self.G2 = GaussianKernel2d(kernel_size, self.sigma2, False)

    @property
    def kernel(self):
        return self.generate_kernel()

    def generate_kernel(self):
        kernel = self.G1.kernel - self.G2.kernel
        if self.normalize:
            kernel /= np.sum(np.abs(kernel))
        return kernel

    def generate_torch_kernel(self):
        return torch.Tensor(self.generate_kernel())

    def convolve(self, image):
        import torch
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = torch.Tensor(image).unsqueeze(0).unsqueeze(0)
            else:
                image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
                image = torch.Tensor(image).unsqueeze(0)

        conv = torch.nn.Conv2d(1, 1, self.kernel_size, bias=False)
        conv.weight.data = torch.Tensor(self.kernel).unsqueeze(0).unsqueeze(0)
        output = conv(image)
        return output.detach().cpu().numpy()[0, ...]

    def edges(self, image, show_image=False):

        filtered_image = self.convolve(image)[0]
        edgy = np.zeros_like(filtered_image)
        row, col = filtered_image.shape[0], filtered_image.shape[1]
        for i in range(1, row-1):
            for j in range(1, col-1):
                if filtered_image[i, j] > 0 and (filtered_image[i-1, j] <= 0 or filtered_image[i+1, j] <= 0 or filtered_image[i, j-1] <= 0 or filtered_image[i, j+1] <= 0):
                    edgy[i, j] = 255

        if show_image:
            from matplotlib import pyplot as plt
            from matplotlib import cm
            # breakpoint()
            plt.figure()
            plt.imshow(edgy, cmap="gray")
            plt.show()
        return edgy
