import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import os

from ..models.custom_layers import GaborConv2d, DReLU, DoGLayer, DoGLowpassLayer, DoG_LP_Layer
from ..read_datasets import imagenette
from .parameters import get_arguments
from ..utils import rgb2yuv, yuv2rgb, per_image_standardization, rgb2gray, DifferenceOfGaussian2d


def plot_image(args, image, image_name, image_index=0, colored=True, show_image=False):
    plt.figure()
    norm = plt.Normalize(vmin=image.min(), vmax=image.max())
    image = norm(image)
    if colored:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')
    plt.title(image_name)
    image_folder = args.directory + "figures/" + "image_" + str(image_index) + "/"
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)
    plt.savefig(image_folder + image_name)
    if show_image:
        plt.show()


def plot_image_histogram(args, image, image_name, image_index=0, show_image=False):
    plt.figure()
    plt.hist(image.ravel(), bins=256, range=(image.min(), image.max()), fc='k', ec='k')
    image_folder = args.directory + "figures/" + "image_" + str(image_index) + "/"
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)
    plt.savefig(image_folder + image_name)
    if show_image:
        plt.show()


def plot_filters(args, filters, image_name, show_image=False):
    num_filters = filters.shape[0]
    row_num = int(np.sqrt(num_filters))
    while num_filters % row_num != 0:
        row_num -= 1
    fig, axs = plt.subplots(row_num, num_filters // row_num)
    for ind, ax in enumerate(axs.flat):
        ax.imshow(filters[ind, 0])
        ax.axis('off')
    image_folder = args.directory + "figures/"
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)
    plt.savefig(image_folder + image_name + ".pdf")

    filters = np.moveaxis(filters, [1, 2, 3], [-1, 1, 2])
    norm = plt.Normalize(vmin=filters.min(), vmax=filters.max())
    fig, axs = plt.subplots(row_num, num_filters // row_num)
    for ind, ax in enumerate(axs.flat):
        img = norm(filters[ind])
        ax.imshow(img)
        ax.axis('off')
    image_folder = args.directory + "figures/"
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)
    plt.savefig(image_folder + image_name + "_rgb.pdf")
    if show_image:
        plt.show()
    # breakpoint()


def gabor_experiments(num_filters, in_channels):

    gabor_layer = GaborConv2d(in_channels=in_channels,
                              out_channels=num_filters, kernel_size=(11, 11))
    gabor_layer.calculate_weights()

    plot_filters(args, gabor_layer.conv_layer.weight.detach().numpy(),
                 "gabor_filters", show_image=False)

    return gabor_layer


def read_images(args):

    train_loader, test_loader = imagenette(args)
    images, labels = test_loader.__iter__().next()
    images_gray = rgb2gray(images, channel_dimension=1)

    return images, images_gray, labels


def center_surround_experiments(args, images, epsilon=8.0/255, image_index=0):

    laplacian = -np.array([[[0, 1, 0], [1, -4, 1], [0, 1, 0]]])
    low_pass = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])/2.
    DoG = DifferenceOfGaussian2d(kernel_size=3, sigma1=0.5, sigma2=0.8, normalize=True)
    dog = np.concatenate((DoG.kernel, DoG.kernel, DoG.kernel))

    cs_lp_weights = np.concatenate((laplacian, low_pass, low_pass))

    cs_lp_filters = torch.nn.Conv2d(in_channels=3, out_channels=3, stride=1,
                                    kernel_size=3, padding=1, groups=3)
    cs_lp_filters_deconv = torch.nn.ConvTranspose2d(
        in_channels=3, out_channels=3, stride=1, kernel_size=3, padding=1, groups=3)

    cs_lp_filters.weight.data = torch.Tensor(cs_lp_weights).unsqueeze(1)
    cs_lp_filters_deconv.weight.data = torch.Tensor(cs_lp_weights).unsqueeze(1)

    out = cs_lp_filters(images)
    plot_image(args, out.permute(0, 2, 3, 1).detach().numpy()[
        image_index], "center_surround_output_yuv.pdf", image_index=image_index, colored=True)

    out = yuv2rgb(out)
    plot_image(args, out.permute(0, 2, 3, 1).detach().numpy()[
        image_index], "center_surround_output_rgb.pdf", image_index=image_index, colored=True, show_image=False)
    out = rgb2yuv(out)

    out = DReLU(out, filters=cs_lp_filters.weight, epsilon=epsilon)
    out = cs_lp_filters_deconv(out)

    out = yuv2rgb(out)
    plot_image(args, out.permute(0, 2, 3, 1).detach().numpy()[
        image_index], "center_surround_recons_output.pdf", image_index=image_index, colored=True, show_image=False)

    plot_image_histogram(args, out.permute(0, 2, 3, 1).detach().numpy()[
        image_index], "hist_center_surround_recons_output.pdf", image_index=image_index)

    plot_image(args, out.detach().numpy()[image_index, 0],
               "center_surround_drelu_output.pdf", image_index=image_index, colored=False)
    plot_image_histogram(args, out.detach().numpy()[image_index, 0],
                         "hist_center_surround_drelu_output.pdf", image_index=image_index)

    # added = 0.1 * images_gray + out
    # # added = per_image_standardization(added)
    # plot_image(args, added.detach().numpy()[image_index, 0],
    #            "center_surround_plus_original.pdf", image_index=image_index, colored=False)
    # plot_image_histogram(args, added.detach().numpy()[image_index, 0],
    #                      "hist_center_surround_plus_original.pdf", image_index=image_index)

    # plt.show()


def dog_experiments(args, images, epsilon=8.0/255, image_index=0):

    # dog = DoGLowpassLayer()
    dog = DoG_LP_Layer(beta=8.0)
    out = dog(images)

    plot_image(args, out.permute(0, 2, 3, 1).detach().numpy()[
        image_index], "dog_lp_output.pdf", image_index=image_index, colored=True, show_image=False)

    plot_image_histogram(args, out.permute(0, 2, 3, 1).detach().numpy()[
        image_index], "hist_dog_lp_output.pdf", image_index=image_index)


args = get_arguments()
images, images_gray, labels = read_images(args)
images_yuv = rgb2yuv(images)
epsilon = 8.0/255

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
breakpoint()

for image_index in range(10):
    # plot_image(args, images.permute(0, 2, 3, 1).numpy()[
    #            image_index], "original_image.pdf", image_index=image_index, colored=True, show_image=False)
    # plot_image_histogram(args, images.permute(0, 2, 3, 1).numpy()[
    #     image_index], "hist_original_image.pdf", image_index=image_index)
    # plot_image(args, images_gray.detach().numpy()[
    #            image_index, 0], "original_gray_image.pdf", image_index=image_index, colored=False)
    # plot_image_histogram(args, images_gray.detach().numpy()[
    #     image_index, 0], "hist_original_gray_image.pdf", image_index=image_index)

    dog_experiments(args, images, epsilon, image_index=image_index)

# gabor_layer = gabor_experiments(num_filters=96, in_channels=3)

# gabor_out = gabor_layer(images)
# drelu_gabor = DReLU(gabor_out, filters=gabor_layer.conv_layer.weight, epsilon=epsilon)

# for image_index in range(10):
#     for filter_index in range(10):
#         plot_image(args, drelu_gabor.detach().numpy()[image_index, filter_index],
#                    "drelu_gabor_output_{}.pdf".format(filter_index), image_index=image_index, colored=False)
