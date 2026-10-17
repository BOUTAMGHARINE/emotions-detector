# Emotion Detector

## 1. Project Overview

This project implements a Convolutional Neural Network (CNN) to classify facial expressions into seven emotion categories using the FER2013 dataset.

The project includes:
- Image preprocessing and dataset exploration.
- CNN architecture design and training.
- Model evaluation using accuracy, loss, and classification metrics.
- Training visualization using learning curves and TensorBoard.
- Emotion prediction on a recorded video.
- Real-time emotion prediction using a webcam.

The objective is to achieve a test accuracy greater than 60% while understanding the complete image classification pipeline.

## 2. Project Structure

```text
project/
├── data/
│   ├── train.csv
│   └── test_with_emotions.csv
├── requirements.txt
├── README.md
├── results/
│   ├── model/
│   │   ├── learning_curves.png
│   │   ├── tensorboard.png
│   │   ├── final_emotion_model_arch.txt
│   │   └── final_emotion_model.keras
│   └── preprocessing_test/
│       ├── image0.png
│       ├── image1.png
│       ├── ...
│       └── input_video.mp4
└── scripts/
    ├── validation_loss_accuracy.py
    ├── predict_live_stream.py
    ├── predict.py
    ├── preprocess.py
    └── train.py
```

The `results/model/` directory contains the trained model, its architecture documentation, and training visualizations.

The `results/preprocessing_test/` directory contains the recorded video and sample preprocessed face images.

## 3. Dataset

The project uses the FER2013 facial expression dataset.

Each image is represented by 2,304 grayscale pixel values, corresponding to an image of 48 × 48 pixels.

The dataset contains seven emotion classes:

| Label | Emotion |
|---:|---|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

The training dataset contains 28,709 images. The classes are imbalanced, with considerably fewer examples of `Disgust` than `Happy`.

This imbalance is important when interpreting the model's performance: overall accuracy alone does not reveal whether all emotions are classified equally well.

## 4. Environment Setup

### Requirements

The project uses Python and the following libraries:

- TensorFlow / Keras for CNN construction, training, and inference.
- NumPy for numerical operations and image arrays.
- Pandas for loading and exploring CSV datasets.
- scikit-learn for evaluation metrics.
- Matplotlib for visualizations.
- OpenCV (`cv2`) for face detection, image processing, and webcam access.

Install the dependencies from the project root:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Make sure `requirements.txt` contains all required dependencies and the versions used to test the project.

## 5. Data Preprocessing

The CNN expects images with the following shape:

```text
(48, 48, 1)
```

The preprocessing pipeline performs these operations:

1. Read the pixel values from the CSV file.
2. Convert the pixel string into a numerical array.
3. Reshape the array into a 48 × 48 grayscale image.
4. Normalize pixel values from the range [0, 255] to [0, 1].
5. Add a channel dimension for the grayscale image.

Normalization is performed using:

```python
image = image.astype(np.float32) / 255.0
```

This gives the model inputs in a consistent numerical range.

### Video and Webcam Preprocessing

For video-based prediction, OpenCV captures frames and a Haar Cascade classifier detects faces.

Each detected face is cropped from the grayscale frame, resized to 48 × 48 pixels, normalized, and reshaped to match the CNN input format.

The model receives an input with shape:

```text
(1, 48, 48, 1)
```

The first dimension represents the batch size: one face is processed at a time.

Preprocessed face images can be saved in `results/preprocessing_test/` for inspection.

## 6. CNN Architecture

A Convolutional Neural Network is suitable for image classification because it learns spatial features directly from image pixels.

The model uses convolutional layers to extract increasingly complex visual features, pooling layers to reduce spatial dimensions, and regularization to help control overfitting.

### Main Components

**Convolutional layers (`Conv2D`)**

Convolutional filters learn local visual patterns, such as edges, contours, and facial features. Deeper layers can combine these patterns into more complex representations.

**ReLU activation**

ReLU introduces non-linearity, allowing the network to learn relationships that cannot be represented by a simple linear transformation.

**Batch normalization**

Batch normalization helps stabilize the distribution of activations during training.

**Max pooling**

Max pooling reduces the spatial dimensions of feature maps while retaining prominent activations.

**Dropout**

Dropout randomly disables a fraction of activations during training. This regularization technique helps reduce dependence on particular neurons and can limit overfitting.

**Output layer**

The final layer produces seven values, one for each emotion class. A softmax activation converts these values into a probability distribution.

The predicted emotion is the class with the highest output probability.

The exact layer configuration and parameter counts are documented in:

```text
results/model/final_emotion_model_arch.txt
```

## 7. Training and Overfitting

The model is trained using the training dataset. A validation subset is used to monitor generalization during training.

The validation split is created from the training data; the independent test dataset should not be used to fit the model.

The training process monitors:
- Training loss.
- Validation loss.
- Training accuracy.
- Validation accuracy.

Learning curves are saved in:

```text
results/model/learning_curves.png
```

### Interpreting Learning Curves

- If training and validation loss decrease together, the model is learning features that also help on unseen validation examples.
- If training loss continues to decrease while validation loss increases, the model may be overfitting.
- If both losses remain high, the model may be underfitting or require improved training.

Early stopping can prevent unnecessary training once validation performance stops improving. The actual stopping strategy depends on the callbacks configured in `train.py`.

TensorBoard can be used to monitor training metrics and compare experiments. The project's TensorBoard screenshot is stored in:

```text
results/model/tensorboard.png
```

## 8. Model Evaluation

The trained model is evaluated on `test_with_emotions.csv`, which contains the pixel data and the corresponding emotion labels.

Run the evaluation script from the project root:

```bash
python ./scripts/validation_loss_accuracy.py
```

The evaluation includes:
- Test accuracy.
- Test loss.
- Precision, recall, and F1-score for each emotion.

The current reported test accuracy is approximately **62%**. Replace this value with the exact result from the final evaluation if it changes.

The project's target is a test accuracy greater than 60%.

With seven classes, uniform random guessing would yield approximately 14.29% accuracy. FER2013 is challenging because facial expressions can be subtle, ambiguous, and visually similar.

Accuracy should be interpreted together with per-class metrics because the dataset is imbalanced.

## 9. Prediction on a Recorded Video

The `predict.py` script detects faces in a recorded video and predicts an emotion for each extracted face.

Run it from the project root:

```bash
python ./scripts/predict.py
```

The script:
1. Loads the trained CNN.
2. Reads and preprocesses the video.
3. Detects faces in sampled frames.
4. Predicts an emotion for each detected face.
5. Prints the predicted class and its probability.

The video path and model path must match the files present in the project.

The prediction probability is the model's estimated probability for the selected class. It is not a guarantee that the prediction is correct.

## 10. Real-Time Webcam Prediction

The `predict_live_stream.py` script performs facial emotion classification using a webcam.

Run it from the project root:

```bash
python ./scripts/predict_live_stream.py
```

The program:
1. Loads the trained model once.
2. Opens the webcam using OpenCV.
3. Captures video frames for at least 20 seconds.
4. Detects faces using Haar Cascade.
5. Converts each detected face to grayscale.
6. Resizes the face to 48 × 48 pixels.
7. Normalizes the pixels and adds the required dimensions.
8. Predicts the emotion and its probability.
9. Displays the result over the webcam image.
10. Saves at least 20 detected, resized face images as PNG files under
    `results/preprocessing_test/webcam/`, in a unique capture subfolder.

The webcam closes automatically when both the 20-second minimum and 20 saved
images have been reached. Press `q` to stop early after 20 seconds; an early
stop before all 20 images are saved is reported as an incomplete capture.

### Recorded Video Fallback

If the webcam is unavailable, a recorded video can be used to test the detection and prediction pipeline. The video must be long enough to satisfy the audit requirement of at least 20 seconds when that requirement is being demonstrated.

The recorded-video fallback should be tested separately from the live webcam script.

## 11. Understanding the Model's Limitations

This project is an educational facial expression classifier, not a reliable system for determining a person's actual emotional state.

Its predictions can be affected by:
- Lighting conditions.
- Face orientation and movement.
- Image quality.
- Class imbalance.
- Differences between the training dataset and real-world images.

A model may predict an emotion with high confidence and still be wrong. Facial expression classification should not be treated as a definitive assessment of someone's feelings or mental health.

## 12. Final Checklist

Before submission, verify the following:

- [ ] All required files and directories are present.
- [ ] `requirements.txt` contains the necessary dependencies and tested versions.
- [ ] The trained model is saved at the expected path.
- [ ] The architecture document describes the chosen CNN and previous iterations.
- [ ] Learning curves and TensorBoard evidence are available.
- [ ] The model is trained using training data only, with validation used for monitoring.
- [ ] The test accuracy is measured on the labeled test dataset.
- [ ] `predict.py` runs without errors and prints the test accuracy if required by the subject.
- [ ] `predict_live_stream.py` runs for at least 20 seconds and saves at least
      20 preprocessed face images in `results/preprocessing_test/webcam/`.
- [ ] The live webcam pipeline displays emotion predictions and probabilities.
- [ ] A recorded video fallback is available if the webcam cannot be used.
- [ ] The learner can explain the architecture, preprocessing, evaluation metrics, and overfitting without reading the code.

## Conclusion

This project demonstrates the complete workflow of an image classification task: data preprocessing, CNN training, validation, test evaluation, and inference on recorded and live video streams.

The main objective is not only to obtain a test accuracy above the required threshold, but also to understand how the model learns visual features, how its performance is evaluated, and why its predictions have limitations.
