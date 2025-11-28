# Heart Disease Prediction Web App

A Streamlit-based web application for predicting heart disease using machine learning models.

## Features

- **Multiple ML Models**: Choose between Logistic Regression, XGBoost, or SVC
- **Interactive Interface**: User-friendly input forms for all medical parameters
- **Real-time Predictions**: Instant predictions with confidence scores
- **Visual Feedback**: Clear indication of prediction results with color-coded alerts

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure you have trained models in the project directory:

   - `LogisticRegression`
   - `XGBoostClassifier`
   - `SVC`

   If you don't have these models, run the `main.ipynb` notebook first to train them.

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

3. Enter patient information in the input fields

4. Select a model from the sidebar

5. Click "Predict Heart Disease" to get the prediction

## Input Features

- **Age**: Patient's age in years
- **Sex**: Gender (Female/Male)
- **Chest Pain Type**: Type of chest pain experienced (0-3)
- **Resting Blood Pressure**: Blood pressure at rest (mm Hg)
- **Serum Cholesterol**: Cholesterol level (mg/dl)
- **Fasting Blood Sugar**: Whether fasting blood sugar > 120 mg/dl
- **Resting ECG**: Electrocardiographic results
- **Maximum Heart Rate**: Maximum heart rate achieved during exercise
- **Exercise Induced Angina**: Whether exercise causes chest pain
- **ST Depression**: ST depression induced by exercise
- **Slope**: Slope of peak exercise ST segment
- **Number of Major Vessels**: Vessels colored by fluoroscopy (0-4)
- **Thalassemia**: Type of blood disorder

## Model Information

The app uses pre-trained models from the main analysis notebook:

- **Logistic Regression**: Linear classification model
- **XGBoost Classifier**: Gradient boosting ensemble method
- **SVC**: Support Vector Classifier with optimized hyperparameters

## Medical Disclaimer

This application is for educational and informational purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for proper medical evaluation.

## Project Structure

```
.
├── app.py                  # Streamlit web application
├── main.ipynb             # Model training notebook
├── heart.csv              # Dataset
├── LogisticRegression     # Trained model
├── XGBoostClassifier      # Trained model
├── SVC                    # Trained model
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## License

This project is for educational purposes.
