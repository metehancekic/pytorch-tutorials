import torch
import numpy as np


def rgb2yuv(images, channel_dimension=1):
    rgb_to_yuv_matrix = np.array(
        [[0.299, 0.587, 0.114], [-0.147, -0.289, 0.436], [0.615, -0.515, -0.1]])
    new_images = images.clone()
    if channel_dimension == 1:
        new_images[:, 0, :, :] = rgb_to_yuv_matrix[0, 0] * images[:, 0, :, :] + \
            rgb_to_yuv_matrix[0, 1] * images[:, 1, :, :] + \
            rgb_to_yuv_matrix[0, 2] * images[:, 2, :, :]
        new_images[:, 1, :, :] = rgb_to_yuv_matrix[1, 0] * images[:, 0, :, :] + \
            rgb_to_yuv_matrix[1, 1] * images[:, 1, :, :] + \
            rgb_to_yuv_matrix[1, 2] * images[:, 2, :, :]
        new_images[:, 2, :, :] = rgb_to_yuv_matrix[2, 0] * images[:, 0, :, :] + \
            rgb_to_yuv_matrix[2, 1] * images[:, 1, :, :] + \
            rgb_to_yuv_matrix[2, 2] * images[:, 2, :, :]
        return new_images
    elif channel_dimension == 3:
        new_images[:, 0, :, :] = rgb_to_yuv_matrix[0, 0] * images[:, :, :, 0] + rgb_to_yuv_matrix[0,
                                                                                                  1] * images[:, :, :, 1] + rgb_to_yuv_matrix[0, 2] * images[:, :, :, 2]
        new_images[:, 1, :, :] = rgb_to_yuv_matrix[1, 0] * images[:, :, :, 0] + rgb_to_yuv_matrix[1,
                                                                                                  1] * images[:, :, :, 1] + rgb_to_yuv_matrix[1, 2] * images[:, :, :, 2]
        new_images[:, 2, :, :] = rgb_to_yuv_matrix[2, 0] * images[:, :, :, 0] + rgb_to_yuv_matrix[2,
                                                                                                  1] * images[:, :, :, 1] + rgb_to_yuv_matrix[2, 2] * images[:, :, :, 2]
        return new_images
    else:
        raise NotImplementedError


def rgb2yrgb(image):

    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    return torch.stack([y, r, g, b], -3)


def rgb2y(image):

    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 0.299 * r
    return y.unsqueeze(1)


def yuv2rgb(images, channel_dimension=1):
    rgb_to_yuv_matrix = np.array(
        [[1., 0., 1.14], [1, -0.395, -0.581], [1., 2.032, 0.]])
    new_images = images.clone()
    if channel_dimension == 1:
        new_images[:, 0, :, :] = rgb_to_yuv_matrix[0, 0] * images[:, 0, :, :] + \
            rgb_to_yuv_matrix[0, 1] * images[:, 1, :, :] + \
            rgb_to_yuv_matrix[0, 2] * images[:, 2, :, :]
        new_images[:, 1, :, :] = rgb_to_yuv_matrix[1, 0] * images[:, 0, :, :] + \
            rgb_to_yuv_matrix[1, 1] * images[:, 1, :, :] + \
            rgb_to_yuv_matrix[1, 2] * images[:, 2, :, :]
        new_images[:, 2, :, :] = rgb_to_yuv_matrix[2, 0] * images[:, 0, :, :] + \
            rgb_to_yuv_matrix[2, 1] * images[:, 1, :, :] + \
            rgb_to_yuv_matrix[2, 2] * images[:, 2, :, :]
        return new_images
    elif channel_dimension == 3:
        new_images[:, 0, :, :] = rgb_to_yuv_matrix[0, 0] * images[:, :, :, 0] + rgb_to_yuv_matrix[0,
                                                                                                  1] * images[:, :, :, 1] + rgb_to_yuv_matrix[0, 2] * images[:, :, :, 2]
        new_images[:, 1, :, :] = rgb_to_yuv_matrix[1, 0] * images[:, :, :, 0] + rgb_to_yuv_matrix[1,
                                                                                                  1] * images[:, :, :, 1] + rgb_to_yuv_matrix[1, 2] * images[:, :, :, 2]
        new_images[:, 2, :, :] = rgb_to_yuv_matrix[2, 0] * images[:, :, :, 0] + rgb_to_yuv_matrix[2,
                                                                                                  1] * images[:, :, :, 1] + rgb_to_yuv_matrix[2, 2] * images[:, :, :, 2]
        return new_images
    else:
        raise NotImplementedError


def rgb2gray(images, channel_dimension=1):
    if channel_dimension == 1:
        return 0.2126*images[:, 0:1, :, :] + 0.7152*images[:, 1:2, :, :] + 0.0722*images[:, 2:3, :, :]
    elif channel_dimension == 3:
        return 0.2126*images[:, :, :, 0:1] + 0.7152*images[:, :, :, 1:2] + 0.0722*images[:, :, :, 2:3]
    else:
        raise NotImplementedError


def per_image_standardization(x):
    x_min, _ = torch.min(x.view(x.size(0), -1), dim=1, keepdim=True)
    x_max, _ = x.view(x.size(0), -1).max(dim=1, keepdim=True)
    for i in range(2):
        x_min = x_min.unsqueeze(1)
        x_max = x_max.unsqueeze(1)
    x = (x-x_min)/(x_max-x_min)
    return x


def show_image(image, name="new_image", show=False, colored=True):
    plt.figure()
    norm = plt.Normalize(vmin=image.min(), vmax=image.max())
    image = norm(image)
    if colored:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')
    plt.title(name)
    plt.savefig(name)
    if show:
        plt.show()
