# Speech Emotion Recognition (SER)

This repository contains a Speech Emotion Recognition (SER) project leveraging a CNN+LSTM model to classify emotions from audio recordings. The project is built to process speech data, extract relevant features, and predict the emotional tone conveyed in the speech.

## Project Overview

Speech Emotion Recognition is an essential area in affective computing and human-computer interaction. By analyzing the emotional tone in speech, this project aims to enhance applications like virtual assistants, customer service bots, and mental health monitoring tools.

Key features of this project include:
- Use of Convolutional Neural Networks (CNN) for feature extraction.
- Integration of Long Short-Term Memory (LSTM) networks to capture temporal dependencies in audio.
- Preprocessing techniques for handling and augmenting speech data.
- Evaluation of model performance with accuracy and classification reports.

## Dataset

This project utilizes:
1. **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
2. **TESS (Toronto Emotional Speech Set)**

These datasets consist of audio files labeled with various emotions, such as happiness, anger, sadness, fear, and neutral.

## Methodology

1. **Data Preprocessing:**
   - Audio signal processing using Librosa for feature extraction (e.g., Mel-Frequency Cepstral Coefficients).
   - Data normalization and augmentation for balanced training.

2. **Model Architecture:**
   - CNN layers for spatial feature extraction from audio spectrograms.
   - LSTM layers to learn temporal patterns in sequential data.
   - Fully connected layers for classification.

3. **Evaluation Metrics:**
   - Confusion matrix for visualizing classification performance.
   - Precision, recall, and F1-score for detailed analysis.

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/sriharshan136/Speech-Emotion-Recognition-CNN-LSTM.git
   cd speech-emotion-recognition
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python train_model.py
   ```

4. Test the model:
   ```bash
   python test_model.py --audio_path /path/to/audio/file
   ```

## Results

The model achieves high accuracy in classifying emotions like happiness, sadness, and anger. Detailed evaluation metrics and confusion matrix results are included in the notebook.

## Contributions

Contributions are welcome! Feel free to open an issue or submit a pull request for improvements or additional features.

## Acknowledgments

- The RAVDESS and TESS datasets for providing high-quality labeled audio data.
- Open-source libraries like Librosa, TensorFlow, and scikit-learn for enabling seamless implementation.

## License

This project is licensed under the MIT License. See the LICENSE file for details.



