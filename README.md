## Face identity detection model

### Description
> This model was created as part of a school project centered around proposing a marketing data concept. The objective of the course was to present an innovative idea within the realm of data methods. Our application is specifically designed to streamline the sharing of event photos, with a primary focus on ensuring user privacy. Leveraging advanced facial recognition algorithms, the system can accurately identify individuals in event photos and selectively facilitate their sharing based on user-defined preferences.

### Table of Contents
- [Features](#features)
- [Dependencies](#dependencies)
- [Data Source](#data-source)
- [Model Training](#model-training)
  
### Features
1. **Face Detection**: Utilizes MTCNN to locate faces within images and returns coordinates.
2. **Face Recognition**: Applies FaceNet to extract faces into feature vectors and compute distances between vectors.
3. **Unsupervised Clustering**: Implements KNN to create clusters of similar faces, enabling identity grouping.
4. **Scalability**: Designed to handle large datasets efficiently for robust identity clustering.

### Dependencies
> This is a list of all dependencies used for the model training.
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
> Data source is not provided since the model was trained of private images. A first file called `photos-private `contains images captured during a private event. A second file called `photos-perso` contains identity images of people that participated to the private event.

### Model Training
1. Data Preparation: Organizing the dataset with labeled images for each identity. Ensure a balanced distribution of identities for effective training.
2. Face Detection and Feature Extraction: Utilizing the MTCNN algorithm to detect faces in images and FaceNet to extract feature vectors. This process transforms each face into a compact representation in a feature space.
3. Labeling Data: Assigning unique labels to each identity in your dataset. This step is crucial for supervised learning.
4. Normalization: Normalizing the feature vectors using the normalize function from scikit-learn. Normalization ensures that all feature vectors have the same scale, which aids in better convergence during training.
5. K-Nearest Neighbors (KNN) Training: EMploying the KNN algorithm to create a model that can classify new face feature vectors into predefined identity clusters. Adjusting the hyperparameters, such as the number of neighbors (k), based on your dataset characteristics.
6. Save Model: Once training is complete, the trained KNN model is saved using a serialization method, such as pickle, to enable reuse without retraining.
