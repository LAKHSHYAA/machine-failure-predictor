# Predictive Maintenance Dashboard

A Streamlit-based web application for predicting machine failure using a pre-trained deep learning model.

- **Live App**: [https://predictive-dashboard.streamlit.app/](https://predictive-dashboard.streamlit.app/)  
- **GitHub Repo**: [https://github.com/LAKHSHYAA/machine-failure-predictor](https://github.com/LAKHSHYAA/machine-failure-predictor)

## Features
- Accepts input for RPM, humidity, vibration, and sensor values.
- Uses a Keras model (`machine_failure_model.h5`) and scaler (`scaler.pkl`) for prediction.
- Provides real-time results in a clean interface.

## Files
- `predict_streamlit.py` – Main dashboard script  
- `prediction.py` – Model and scaler loading  
- `requirements.txt` – Project dependencies  
- `machine_failure_model.h5`, `scaler.pkl` – Model assets  
- `predictive_maintenance.csv` – Sample dataset

## Run Locally
```bash
git clone https://github.com/LAKHSHYAA/machine-failure-predictor.git
cd machine-failure-predictor
pip install -r requirements.txt
streamlit run predict_streamlit.py
