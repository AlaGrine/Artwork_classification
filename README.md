# Artwork classification with PyTorch

## Image classification in PyTorch using Convolutional and Transformer models

<div align="center">
  <img src="https://github.com/AlaGrine/Artwork_classification_in_PyTorch/blob/main/results/fig_Artbench.png" >
</div>

### Table of Contents

1. [Project Overview](#overview)
2. [Installation](#installation)
3. [File Descriptions](#file_descriptions)
4. [Modelling](#modelling)
5. [Results](#results)
6. [Deploy a Gradio demo to HuggingFace Spaces](#gradio_demo)
7. [Acknowledgements](#ack)

## Project Overview <a name="overview"></a>

The aim of this project is to classify artworks using the [artbench](https://github.com/liaopeiyuan/artbench) dataset. The dataset includes 60,000 images of artworks from 10 different artistic styles, including paintings, murals, and sculptures from the 14th to the 21st century. Each style has 5,000 training images and 1,000 test images.

I used PyTorch to create convolutional and transformer-based models. Specifically, I leveraged and fine-tunde the pre-trained [EfficientNet_B2](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b2.html#torchvision.models.efficientnet_b2)-B2 and [ViT_B16](https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html?highlight=vit#torchvision.models.vit_b_16) models.

I replicated the [ViT paper](https://arxiv.org/abs/2010.11929) and built the ViT model from scratch to enhance comprehension of the Transformer architecture.

To speed up model training, I used two free Kaggle GPU T4 accelerators.

I also created a [Gradio](https://www.gradio.app/) demo and deployed the app to [HuggingFace Spaces](https://huggingface.co/spaces).

You can find the app [here](https://huggingface.co/spaces/AlaGrine/Artwork_classifier).

## Installation <a name="installation"></a>

This project requires Python 3 and the following Python libraries installed:

`torch` ,`torchvision`, `torchinfo`, `torchmetrics`, `mlxtend`, `pandas`, `numpy`, `sklearn`, `matplotlib`, `wget`, `tarfile`, `gradio`

## File Descriptions <a name="file_descriptions"></a>

The main file of the project is `artwork_classification.ipynb`.

The project folder also contains the following:

- `artbench-10-imagefolder-split` folder: Download the artbench dataset into this folder.
- `results` folder: Includes model metrics and figures.
- `models` folder: Includes fine-tuned EfficientNet_B2 model (.pth file).
- `gardio_demo` folder: Includes the [Gradio](https://www.gradio.app/) demo application.

## Modelling <a name="modelling"></a>

I built the following models:

1. TinyVGG model as described in the [CNN Explainer](https://poloclub.github.io/cnn-explainer/) website.
2. Convolutional-based model: [EfficientNet_B2](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b2.html#torchvision.models.efficientnet_b2) feature extraction on 20% of the data with data augmentation
3. EfficientNet_B2: Fine-tuning on the full dataset.
4. Transformer-based model: [ViT_B16](https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html?highlight=vit#torchvision.models.vit_b_16) feature extraction on 20% of the data with data augmentation
5. ViT_B16: Fine-tuning on the full dataset.

## Results<a name="results"></a>

<div align="center">
  <img src="https://github.com/AlaGrine/Artwork_classification_in_PyTorch/blob/main/results/fig_inference-speed-vs-accuracy.jpg" >
</div>

The EfficientNet_B2 model outperforms the ViT_B16 model in all performance metrics. It achieves the highest accuracy, lowest loss, smallest size, and shortest prediction time per image.

## Deploy a Gradio demo to HuggingFace Spaces <a name="gradio_demo"></a>

<div align="center">
  <img src="https://github.com/AlaGrine/Artwork_classification_in_PyTorch/blob/main/results/fig_gradio_demo.png" >
</div>

To deploy the [Gradio](https://www.gradio.app/) demo to [HuggingFace Spaces](https://huggingface.co/spaces), follow these steps:

1.  Create a new space (ie. code repository). Space name = [SPACE_NAME].
2.  Select Gradio as the Space SDK and CPU basic (free) as Space hardware.

Then, follow the standard git workflow:

3.  Clone the repo locally: `git clone https://huggingface.co/spaces/[USERNAME]/[SPACE_NAME]`
4.  Copy the contents of `gradio_demo` folder to the `clonded repo` folder.
5.  Passwords are no longer accepted as a way to authenticate command-line Git operations. You need to use a personal access token as explained [here](https://huggingface.co/blog/password-git-deprecation).

            `git remote set-url origin https://[USERNAME]:[TOKEN]@huggingface.co/spaces/[USERNAME]/[SPACE_NAME]`

6.  `git add .`
7.  `git commit -m "first commit"`
8.  `git push`

## Acknowledgements <a name="ack"></a>

Credit must be given to the authors of the [artbench](https://github.com/liaopeiyuan/artbench) dataset.
