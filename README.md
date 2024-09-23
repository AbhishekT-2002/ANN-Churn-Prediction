# Customer Churn Prediction Using ANN

This project predicts whether a customer is likely to churn (leave the service) based on several input features using an Artificial Neural Network (ANN) model. The app is built using Streamlit for an interactive user interface.

## Features

The model uses the following features to predict customer churn:

- **Geography**: The location of the customer (one-hot encoded).
- **Gender**: The gender of the customer (encoded).
- **Age**: The age of the customer.
- **Credit Score**: Customer's credit score.
- **Balance**: The current balance of the customer.
- **Tenure**: The number of years the customer has been with the company.
- **Number of Products**: The number of products the customer uses.
- **Has Credit Card**: Whether the customer has a credit card (0 or 1).
- **Is Active Member**: Whether the customer is an active member (0 or 1).
- **Estimated Salary**: The estimated salary of the customer.

## Prerequisites

Ensure you have the following installed:

- Python 3.x
- TensorFlow
- Streamlit
- Pandas
- Scikit-learn
- Pickle (for loading encoders and scaler)

## Model and Files

You will need the following files to run the app locally:

1. `model.h5`: The pre-trained ANN model.
2. `gender_encoder_le.pkl`: Label encoder for the `Gender` feature.
3. `onehot_encoder_geo.pkl`: One-hot encoder for the `Geography` feature.
4. `sscaler.pkl`: Standard scaler to scale the input features.

## How to Run Locally

1. Clone the repository or download the project files.
2. Place all necessary model and encoder files in the same directory as the Streamlit app.
3. Install the required dependencies by running:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app using the following command:

    ```bash
    streamlit run app.py
    ```

5. Open the URL provided in the terminal to access the web app.

## Deployment

The model has been deployed and can be accessed at the following link:

[Customer Churn Prediction App](https://ann-churn-prediction-69.streamlit.app/)

## Usage

- Select the customer details such as `Geography`, `Gender`, `Age`, `Credit Score`, etc.
- Once all fields are filled, the model will predict whether the customer is likely to churn or stay.
- If the churn probability is greater than 0.5, the customer is predicted to churn. Otherwise, they are predicted to stay.

## Files in the Project

- `app.py`: The main Streamlit app script.
- `model.h5`: The saved ANN model.
- `gender_encoder_le.pkl`: Gender label encoder.
- `onehot_encoder_geo.pkl`: Geography one-hot encoder.
- `sscaler.pkl`: Standard scaler for feature scaling.

## License

This project is licensed under the MIT License.

## Acknowledgements

This project was built using:

- [Streamlit](https://streamlit.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [Scikit-learn](https://scikit-learn.org/)
