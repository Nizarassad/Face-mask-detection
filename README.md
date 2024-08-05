
# Face Mask Detection and Classification in Real-Time

<p align="center">
  <img src="https://drive.google.com/uc?id=1fD0fEHzppnu_MLXEjyh07JmNHjKSH0JA" alt="Black Friday Analysis">
</p>

The goal of this project is to develop a Face Covering Detection (FCD) computer vision system. The system is capable of detecting whether a person is wearing a face mask correctly, incorrectly, or not wearing a mask at all. The three classes for classification are:

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
- **Deep Learning Framework**: TensorFlow/Keras and PyTorch
- **Computer Vision**: OpenCV
- **Webcam Integration**: OpenCV for capturing real-time video feed
- **Pre-trained Models**: Transfer learning using models like MobileNetV2 or ResNet

## Dataset

The dataset can be found in the link below.
- https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

## Models

In this study, four machine learning combinations were compared based on their accuracy, precision, recall, inference time, and model size:

1. **CNN+HOG**
2. **SVM+HOG**
3. **MLP+HOG**
4. **MLP+LBP**

These models were chosen due to their different strengths and weaknesses, such as size differences. HOG and LBP features were extracted from the original images to create a more condensed representation of the images. 

- CNN+HOG model can be downloaded through this link: https://drive.google.com/file/d/165IkA47Tov2M3CfhYvJuQ5mCWGxkhMe1/view?usp=sharing

## Results

### Qualitative Results

We tested our model on images and videos to evaluate its real-world performance. The models should be able to detect face masks in different settings, such as indoor and outdoor environments, and varying lighting conditions.

**Images:**We tested the models on a set of four random images, which included the 3 different classes. In most cases, the models were able to successfully identify the presence and position of face masks, even under challenging lighting conditions. 

**Videos:**We further evaluated the performances of all four models on video files, where each model processed a continuous stream of frames in real-time. The models maintained a consistent level of accuracy throughout the video, accurately detecting face masks despite changes in subject positions, lighting, and background. 



### Quantitative Results

The table below shows the performances of the four models on the test set:

| Model      | Accuracy | Precision | Recall  | Inference Time | Model Size |
|------------|----------|-----------|---------|----------------|------------|
| CNN+HOG    | 85.15%   | 61.70%    | 34.64%  | 1.2629s        | 130.18 MB  |
| SVM+HOG    | 82.10%   | 57.43%    | 56.01%  | 0.2232s        | 9.62 MB    |
| MLP HOG    | 83.62%   | 79.76%    | 52.86%  | 0.0086s        | 2.50 MB    |
| MLP LBP    | 83.62%   | 42.15%    | 40.28%  | 0.0020s        | 1.15 MB    |

## Discussion

Overall, the CNN+HOG model delivered the best performance in terms of accuracy, making it the best choice for the face mask detection system. However, the MLP+LBP model can be more recommended as speed and model size are important.

## References

[1] Wang, L., Lin, Z., Wong, A., "Covid-19 Face Mask Detection Using Deep Learning and Computer Vision Techniques," Image Processing On Line, 10 (2020), pp. 252-262. https://doi.org/10.5201/ipol.2020.288

[2] Elaziz, M. A., Hosny, K. M., Salah, A., Darwish, M. M., Lu, S., & Sahlol, A. T. (2021). Deep Learning for Face Mask Detection in the Era of COVID-19 Pandemic:
