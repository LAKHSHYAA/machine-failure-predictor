# Predictive Maintenance Dashboard
Project Description:
A Streamlit-based web application for predicting machine failure using a pre-trained deep learning model.

Live Application:
https://predictive-dashboard.streamlit.app/

GitHub Repository:
https://github.com/LAKHSHYAA/machine-failure-predictor

Functionality:

Accepts user input for operational parameters including revolutions per minute, humidity, vibration, and sensor readings.

Preprocesses inputs using a saved scaler (scaler.pkl) consistent with the model's training pipeline.

Generates predictions using a TensorFlow/Keras model (machine_failure_model.h5).

Presents results through an interactive Streamlit dashboard.

File Structure:

predict_streamlit.py: Main Streamlit application.

prediction.py: Contains model loading and prediction logic.

machine_failure_model.h5: Trained machine learning model.

scaler.pkl: Data scaler used during training.

predictive_maintenance.csv: Sample dataset for demonstration.

requirements.txt: List of required Python packages.

Setup Instructions:

Clone the repository:
git clone https://github.com/LAKHSHYAA/machine-failure-predictor.git

Navigate to the project directory:
cd machine-failure-predictor

(Optional) Create and activate a virtual environment.

Install dependencies:
pip install -r requirements.txt

Run the application locally:
streamlit run predict_streamlit.py

Dependencies:

streamlit

pandas

numpy

scikit-learn

tensorflow

joblib
