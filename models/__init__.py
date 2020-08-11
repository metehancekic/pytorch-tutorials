"""
	Neural Architectures for image classification
"""

from .resnet import ResNet
from .attention_models import AttentionResNet, ConvolutionalAttentionResNet, SpatialAttentionResNet, ConvolutionalSpatialAttentionResNet
from .vgg import VGG
from .mobilenet import MobileNet
from .mobilenetv2 import MobileNetV2
from .preact_resnet import PreActResNet_wrapper as PreActResNet
from .embedding_classification import MultiModeEmbeddingClassification, MultiModeEmbeddingMaxpooling

__all__ = ["ResNet", "AttentionResNet", "ConvolutionalAttentionResNet",
           "SpatialAttentionResNet", "ConvolutionalSpatialAttentionResNet",
           "VGG", "MobileNet", "MobileNetV2", "PreActResNet",
           "MultiModeEmbeddingClassification"]
