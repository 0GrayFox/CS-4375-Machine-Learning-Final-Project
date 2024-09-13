# CNN Architecture Analysis: LeNet, VGG16, and Modified EfficientNet

## Abstract
This project analyzes three convolutional neural network (CNN) architectures: LeNet, VGG16, and a modified version of EfficientNet. The report compares their performance in terms of loss, accuracy (training, testing, validation), and computational efficiency using standardized datasets (Fashion-MNIST and CIFAR10). LeNet, though simple, struggles with complex images. VGG16 offers higher accuracy but at the cost of resource intensity. Modified EfficientNet strikes a balance, showing scalability and efficiency. Results recommend VGG16 for small-to-medium datasets requiring high accuracy and EfficientNet for complex datasets, though it may overfit smaller datasets. The study underlines the importance of choosing a CNN architecture based on specific needs and constraints.

## Introduction
Deep learning and CNNs have significantly advanced image processing, providing various architectures for specific tasks. CNNs mimic human visual processing and are applied to extract features from images, making them crucial in tasks such as image recognition. This project aims to evaluate and compare the effectiveness of three distinct CNN architectures—LeNet, VGG16, and a modified EfficientNet (B0 version)—to determine the best balance of accuracy and computational efficiency across different tasks.

The study objectives are:
1. **Compare LeNet, VGG16, and EfficientNet on performance and accuracy.**
2. **Assess computation resources and efficiency for resource-constrained environments.**
3. **Provide recommendations for selecting the appropriate CNN architecture based on the dataset and application.**

## Models Overview
- **LeNet**: A simpler architecture designed for low-resolution images. Suitable for small datasets but lacks performance for complex visual tasks.
- **VGG16**: A deeper network offering higher accuracy but is resource-intensive. Best suited for small-to-medium datasets where accuracy is critical.
- **Modified EfficientNet**: Built for efficiency and scalability with a balance of performance and resource usage. It may overfit smaller datasets but excels with complex data.

## Methodology
The models were implemented on two datasets: **Fashion-MNIST** and **CIFAR10**. We conducted a series of experiments to measure performance metrics like accuracy, loss, and computational requirements. The experiments included the following steps:
1. Building and fine-tuning the LeNet and VGG16 models.
2. Modifying the EfficientNet (B0 version) by fine-tuning it to fit our datasets.
3. Using activation functions like ReLU for LeNet and VGG16, and SiLU for EfficientNet.
4. Training the models over 25 epochs to optimize their performance.

### Key Techniques
- **Feature Detection**: Using convolutional layers to apply filters (kernels) on the input image.
- **Activation Functions**: Applying non-linearity with ReLU and SiLU to learn complex patterns.
- **Pooling Layers**: Using max pooling to reduce spatial dimensions and computational load.
- **Backpropagation**: Optimizing filter weights using optimization algorithms to minimize the loss function.

### Data Visualization
Exploratory data analysis (PCA, LDA, t-SNE) was conducted to visualize datasets and understand their distributions. TensorBoard was used to visualize the training progress and performance.

## Results & Discussion
- **LeNet**: Adequate for low-resolution images but falls short on more complex datasets.
- **VGG16**: High accuracy, but with significant resource consumption. Ideal for small-to-medium datasets.
- **EfficientNet (Modified)**: Balanced performance, scalable to various environments, but may overfit smaller datasets.

## Conclusion
The choice of CNN architecture should be based on dataset size, complexity, and available computational resources. VGG16 is best suited for tasks requiring high accuracy on small-to-medium datasets, while modified EfficientNet is ideal for large, complex datasets but may not be optimal for simpler tasks.

## Authors
We both contributed equally to implementing, testing, and analyzing all three CNN architectures.
