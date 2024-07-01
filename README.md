
# Face Mask Detection and Classification in Real-Time

## Project Description

This project aims to develop a real-time face mask detection and classification system using computer vision techniques. The system is capable of detecting whether a person is wearing a face mask correctly, incorrectly, or not wearing a mask at all. The three classes for classification are:

1. **Mask**: The face mask is worn correctly, covering both the nose and mouth.
2. **No Mask**: The face mask is not worn, and the face is fully visible.
3. **Mask Incorrect**: The face mask is worn incorrectly, not covering either the nose or mouth properly.

## Features

- **Real-Time Detection**: Utilizes a webcam or video feed to detect and classify face masks in real-time.
- **Multi-Class Classification**: Distinguishes between three classes: mask, no mask, and mask incorrect.
- **High Accuracy**: Employs deep learning models to ensure high accuracy in detection and classification.
- **User-Friendly Interface**: Simple and intuitive interface for easy use and interaction.
- **Cross-Platform Compatibility**: Runs on various operating systems including Windows, macOS, and Linux.

## Technologies Used

- **Programming Languages**: Python
- **Deep Learning Framework**: TensorFlow/Keras or PyTorch
- **Computer Vision**: OpenCV
- **Webcam Integration**: OpenCV for capturing real-time video feed
- **Pre-trained Models**: Transfer learning using models like MobileNetV2 or ResNet

## Data

The goal of this project is to develop a Face Covering Detection (FCD) computer vision system. A ready-to-use dataset of cropped photographs of faces has been provided, categorized into three classes: appropriately wearing a face mask, not wearing a face mask, or wearing a face mask incorrectly. The dataset includes testing, validation, and training sets.


## Implemented Methods

Machine learning models are required to detect human faces in various positions and lighting conditions and detect the presence and location of face coverings. In this study, four machine learning combinations were compared based on their accuracy, precision, recall, inference time, and model size:

1. **CNN+HOG**
2. **SVM+HOG**
3. **MLP+HOG**
4. **MLP+LBP**

These models were chosen due to their different strengths and weaknesses, such as size differences. HOG and LBP features were extracted from the original images to create a more condensed representation of the images. HOG features record information on gradient orientation, while LBP features encode regional texture patterns. The Standard Scaler was employed to standardize the features, giving features with a mean of 0 and a variance of 1.


## Results

### Qualitative Examples

We tested our model on images and videos to evaluate its real-world performance. The models should be able to detect face masks in different settings, such as indoor and outdoor environments, and varying lighting conditions.

#### Images

We tested the models on a set of four random images, which included individuals wearing masks correctly, incorrectly, and not wearing masks at all. In most cases, the models were able to successfully identify the presence and position of face masks, even under challenging lighting conditions. These results indicate the models' effectiveness and potential applicability in real-world situations. Nevertheless, some refinements and improvements are still necessary to achieve optimal performance across all scenarios.

#### Videos

We further evaluated the performances of all four models on video files, where each model processed a continuous stream of frames in real-time. The models maintained a consistent level of accuracy throughout the video, accurately detecting face masks despite changes in subject positions, lighting, and background. However, some false positive cases were still observed in certain frames. While the general performance of the models is promising, further fine-tuning and optimization may be crucial for deployment in real-world applications where minimizing false positives is essential.

### Quantitative Results

The table below shows the performances of the four models on the test set:

| Model      | Accuracy | Precision | Recall  | Inference Time | Model Size |
|------------|----------|-----------|---------|----------------|------------|
| CNN+HOG    | 85.15%   | 61.70%    | 34.64%  | 1.2629s        | 130.18 MB  |
| SVM+HOG    | 82.10%   | 57.43%    | 56.01%  | 0.2232s        | 9.62 MB    |
| MLP HOG    | 83.62%   | 79.76%    | 52.86%  | 0.0086s        | 2.50 MB    |
| MLP LBP    | 83.62%   | 42.15%    | 40.28%  | 0.0020s        | 1.15 MB    |

## Discussion

The CNN+HOG model achieved the highest accuracy of 85.15%. This model combined the strengths of a deep Convolutional Neural Network (CNN) and the Histogram of Oriented Gradients (HOG) features. However, it had a much larger model size and longer inference time compared to the other models, which may be a limiting factor given that inference speed is very important.

The MLP+LBP model provided a good balance between accuracy, speed, and model size, making it a suitable choice for real-time applications where speed and model size are primary concerns. This model used a simple Multi-Layer Perceptron (MLP) architecture with Local Binary Patterns (LBP) features, which are computationally efficient and invariant to lighting changes.

The SVM+HOG and MLP+HOG models achieved comparable accuracy scores, but the MLP+HOG model had the advantage of higher precision and lower inference time. The SVM+HOG model used a Support Vector Machine (SVM) with HOG features, which is a popular combination for object detection tasks. However, it had a longer inference time compared to the MLP-based models.

Overall, the CNN+HOG model delivered the best performance in terms of accuracy, making it the best choice for the face mask detection system. However, the MLP+LBP model can be more recommended as speed and model size are important.

## References

[1] Wang, L., Lin, Z., Wong, A., "Covid-19 Face Mask Detection Using Deep Learning and Computer Vision Techniques," Image Processing On Line, 10 (2020), pp. 252-262. https://doi.org/10.5201/ipol.2020.288

[2] Elaziz, M. A., Hosny, K. M., Salah, A., Darwish, M. M., Lu, S., & Sahlol, A. T. (2021). Deep Learning for Face Mask Detection in the Era of COVID-19 Pandemic:
