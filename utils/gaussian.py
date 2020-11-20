"""
Gaussian Filter
Difference of Gaussian Filter
"""
import sys
import numpy as np


class GaussianKernel2d(object):
    """docstring for GaussianKernel2d"""

    def __init__(self, kernel_size, sigma, normalize=True):
        super(GaussianKernel2d, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.normalize = normalize

    @property
    def kernel(self):
        return self.generate_kernel()

    def generate_kernel(self):

        x_axis = np.arange(-(self.kernel_size//2), self.kernel_size//2+1)
        y_axis = np.arange(-(self.kernel_size//2), self.kernel_size//2+1)
        x, y = np.meshgrid(x_axis, y_axis)
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        for i, xi in enumerate(x_axis):
            for j, yi in enumerate(y_axis):
                kernel[i, j] = self.gaussian(np.array([xi, yi]), np.array(
                    [0, 0]), np.array([[self.sigma**2, 0.], [0., self.sigma**2]]))
        if self.normalize:
            kernel /= np.sum(kernel)
        return kernel

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


def test():
    np.set_printoptions(threshold=sys.maxsize)

    DoG = DifferenceOfGaussian2d(kernel_size=3, sigma1=0.5, sigma2=0.8, normalize=True)
    print(DoG.kernel)
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])/8
    import skimage
    import matplotlib.pyplot as plt
    cameraman = skimage.data.camera()/255

    filtered_image = DoG.convolve(cameraman)
    plt.figure()
    plt.subplot(121)
    plt.imshow(cameraman, cmap="gray")
    plt.subplot(122)
    plt.imshow(filtered_image[0, ...], cmap="gray")
    # plt.colorbar()
    # plt.show()
    DoG.edges(cameraman, show_image=True)
    breakpoint()

    # G = GaussianKernel2d(kernel_size=3, sigma=0.4, normalize=True)
    # # G.plot()
    # DoG = DifferenceOfGaussian2d(kernel_size=3, sigma1=1.5, normalize=True)
    # # DoG.plot()
    # print(GaussianKernel2d(kernel_size=3, sigma=0.1, normalize=False).kernel)
    # print(GaussianKernel2d(kernel_size=3, sigma=5, normalize=False).kernel)
    # # A = GaussianKernel2d(kernel_size=3, sigma=1.6, normalize=True).kernel - \
    # #     GaussianKernel2d(kernel_size=3, sigma=5, normalize=True).kernel
    # # print(A/np.sum(np.abs(A)))
    # print(DoG.kernel)


def main():
    test()


if __name__ == '__main__':
    main()
