
import torch
from torch import nn
import torchvision

def create_effnetb2_model(num_classes:int=10,
                          seed:int=42,
                          is_TrivialAugmentWide = True,
                          freeze_layers=True):
    """Creates an EfficientNet_B2 model and transforms.

    Args:
        num_classes (int): number of classes in the classifier head, default = 10.
        seed (int): random seed value, default = 42.
        is_TrivialAugmentWide (boolean): Artificially increase the diversity of a training dataset with data augmentation, default = True.
        freeze_layers (boolean): if True, all layers will be frozen.

    Returns:
        effnetb2_model (torch.nn.Module): EfficientNet_B2 feature extractor model.
        effnetb2_transforms (torchvision.transforms): EfficientNet_B2 image transforms.
    """
    # 1. Create EfficientNet_B2 pretrained weights and transforms
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    effnetb2_transforms = weights.transforms()

    if is_TrivialAugmentWide:
      effnetb2_transforms = torchvision.transforms.Compose([
          torchvision.transforms.TrivialAugmentWide(),
          effnetb2_transforms,
      ])

    # 2. Create EfficientNet_B2 model
    effnetb2_model = torchvision.models.efficientnet_b2(weights=weights)

    # 3. Freeze all layers
    if freeze_layers:
        for param in effnetb2_model.parameters():
            param.requires_grad = False

    # 4. Change classifier head
    torch.manual_seed(seed)
    effnetb2_model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes),
    )

    return effnetb2_model, effnetb2_transforms
