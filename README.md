# Multiclass_Animal_Recognition
# Multi-Class Animal Recognition

## Overview
This project implements a multi-class animal recognition system using deep learning. The objective is to identify different animal species from images by leveraging a pre-trained convolutional neural network (CNN) model, MobileNetV2. The workflow involves dataset preprocessing, model training and evaluation, and deploying the model for inference.

## Features
- **Dataset**: Utilizes the "90 Different Animals" dataset available on Kaggle.
- **Transfer Learning**: Fine-tunes the MobileNetV2 model pre-trained on ImageNet.
- **Data Augmentation**: Improves model robustness by applying techniques like rotation, zoom, and flipping during training.
- **Evaluation Metrics**: Provides detailed accuracy and loss metrics with visualizations.
- **Prediction Capability**: Supports real-time predictions from image inputs.

## Prerequisites
- Python 3.7 or later
- GPU (recommended for faster training)
- Libraries:
  - TensorFlow
  - NumPy
  - Matplotlib
  - PIL (Python Imaging Library)

## Installation
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:
   - The "90 Different Animals" dataset is available on [Kaggle](https://www.kaggle.com/iamsouravbanerjee/animal-image-dataset-90-different-animals).
   - Place the dataset in the appropriate directory (e.g., `data/`).

## Dataset Structure
The dataset should follow the structure below:
```
data/
  animals/
    cat/
    dog/
    elephant/
    ...
```

## Usage

### 1. Preprocessing
- Load the dataset and preprocess it for the model. This includes resizing images to the required input shape and splitting them into training and validation sets.
- Example code snippet:
  ```python
  img_size = (224, 224)  # Input size for MobileNetV2
  batch_size = 32
  train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
  train_generator = train_datagen.flow_from_directory('data/animals/', target_size=img_size, batch_size=batch_size)
  ```

### 2. Model Training
- Fine-tune MobileNetV2 by adding custom dense layers for classification.
- Train the model using the preprocessed dataset:
  ```python
  history = model.fit(train_generator, epochs=10, validation_data=val_generator)
  ```

### 3. Evaluation
- Evaluate the model’s performance using metrics like accuracy and loss.
- Visualize the training and validation curves:
  ```python
  plt.plot(history.history['accuracy'], label='Training Accuracy')
  plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
  plt.legend()
  plt.show()
  ```

### 4. Saving the Model
- Save the trained model for later use:
  ```python
  model.save('MC_AR.keras')
  ```

### 5. Prediction
- Load the saved model and use it for inference:
  ```python
  from tensorflow.keras.models import load_model
  from tensorflow.keras.preprocessing import image

  model = load_model('MC_AR.keras')
  def predict_animal(image_path):
      img = image.load_img(image_path, target_size=(224, 224))
      img_array = image.img_to_array(img) / 255.0
      img_array = np.expand_dims(img_array, axis=0)
      predictions = model.predict(img_array)
      return np.argmax(predictions, axis=1)
  ```

## Results
- The model demonstrates high accuracy on the validation dataset.
- Visualizations of training progress, including accuracy and loss curves, are provided.

## Contributing
Contributions are welcome! If you would like to contribute, please:
1. Fork this repository.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments
- Dataset: [90 Different Animals](https://www.kaggle.com/iamsouravbanerjee/animal-image-dataset-90-different-animals)
- Framework: TensorFlow and Keras
- Pre-trained Model: MobileNetV2

