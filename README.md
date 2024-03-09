## Face Identity Detection Algorithm

### Description
> This repository was created as part of a school project. The objective of the course was to present an innovative idea within the realm of data methods. Our application is specifically designed to streamline the sharing of event photos, with a primary focus on ensuring user privacy. Leveraging advanced facial recognition algorithms, the system can accurately identify individuals in event photos and selectively facilitate their sharing based on user-defined preferences. This repository shares the training and testing phase of the algorithm. Pre-trained methods are listed in [Algorithm Process](#algorithm-process).

### Table of Contents
- [Features](#features)
- [Dependencies](#dependencies)
- [Data Source](#data-source)
- [Algorithm Process](#algorithm-process)
- [Script Structure](#script-structure)
  
### Features
1. **Face Detection**: Utilizes MTCNN to locate faces within images and returns coordinates.
2. **Face Recognition**: Applies FaceNet to extract faces into feature vectors and compute distances between vectors.
3. **Unsupervised Clustering**: Implements KNN to create clusters of similar faces, enabling identity grouping.
4. **Scalability**: Designed to handle large datasets efficiently for robust identity clustering.

### Dependencies
- `numpy`
- `matplotlib`
- `mtcnn`
- `Pillow` (PIL)
- `face_recognition`
- `opencv-python` (cv2)
- `scipy`
- `imgaug`
- `scikit-learn`

### Data Source
> Data source is not provided since the model was trained on private images. The first file called `photos-private `contains images captured during a private event. A second file called `photos-perso` contains identity images of people who participated in the private event.

### Algorithm process
1. Data Preparation: Organizing the dataset with labeled images for each identity.
2. Face Detection and Feature Extraction: Utilizing the MTCNN algorithm to detect faces in images and FaceNet to extract feature vectors. This process transforms each face into a compact representation in a feature space.
3. Normalization: Normalizing the feature vectors using the normalize function from scikit-learn. Normalization ensures that all feature vectors have the same scale, which aids in better convergence during training.
4. K-Nearest Neighbors (KNN) Training: Employing the KNN algorithm to create a model that can classify new face feature vectors into predefined identity clusters. Adjusting the hyperparameters, such as the number of neighbors (k), based on your dataset characteristics.
5. Data Augmentation:  Enhancing face detection robustness by applying data augmentation techniques using the `imgaug` library. Common augmentations include rotation, scaling, and flipping, which help the model generalize better to variations in pose and lighting.

### Script Structure
> The script structure is decomposed into 3 parts. In the first one, we prepare the functions for the prediction process of the second and third parts.

1) Define our functions
  a) Functions for loading the images, detecting faces, using MTCNN, extracting vectors
  b) Function to extract face vectors using data augmentation method

2) Predict identities with data augmentation
  a) Extract faces' vectors on identity pictures
  b) Loop on every event pictures
  c) Display all photos' names where identity is detected

3) Predict identities without data augmentation (testing phase)
  a) Extract faces' vectors on identity pictures
  b) Loop on every event pictures
  c) Display the results on the event pictures
