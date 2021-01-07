"""
	Neural Architectures for image classification
"""

from .resnet import ResNet, ResNetEmbedding, ResNetEmbedding2d, ResNetWide
from .wideresnet import WideResNet
from .attention_models import AttentionResNet, ConvolutionalAttentionResNet, SpatialAttentionResNet, ConvolutionalSpatialAttentionResNet
from .vgg import VGG
from .VGG_modified import VGG_modified
from .mobilenet import MobileNet
from .mobilenetv2 import MobileNetV2
from .preact_resnet import PreActResNet_wrapper as PreActResNet
from .embedding_classification import MultiModeEmbeddingClassification, MultiModeEmbeddingMaxpooling
from .efficientnet import EfficientNet
from .lenet import LeNet, LeNet2d
from . import custom_layers

__all__ = ["ResNet", "WideResNet", "AttentionResNet", "ConvolutionalAttentionResNet",
           "SpatialAttentionResNet", "ConvolutionalSpatialAttentionResNet",
           "VGG", "MobileNet", "MobileNetV2", "PreActResNet",
           "MultiModeEmbeddingClassification", "EfficientNet", "LeNet", "LeNet2d", "ResNetEmbedding", "ResNetEmbedding2d", "ResNetWide"]
