# Mobiticket Seat Sales Prediction

## Overview
This project aims to predict the number of seats sold for rides from various towns in Kenya to Nairobi using machine learning models. The goal is to develop a robust model capable of capturing patterns in ticket sales based on historical data, departure times, and external factors such as traffic conditions.

## Dataset
The dataset consists of the following files:
- **Train.csv**: Contains historical ticket sales data, including `ride_id`, `travel_from`, `travel_date`, `departure_time`, `route`, `vehicle_type`, and `number_of_tickets`.
- **Test.csv**: Contains the same features as the training set, except for the `number_of_tickets` column (which is to be predicted).
- **sample_submission.csv**: A template for submitting predictions, with `ride_id` and `number_of_tickets` columns.

## Approach
The project employs machine learning techniques to build a regression model for predicting ticket sales.

## Data Preprocessing
- **Feature Engineering**:
  - Extracted `hour`, `day_of_week`, and `month` from `travel_date`.
  - Categorized `departure_time` into time bins (e.g., morning, afternoon, evening).
  - Added external traffic data (e.g., Uber Movement data for Nairobi traffic trends).
- **Data Cleaning**:
  - Handled missing values using appropriate imputations.
  - Encoded categorical variables using one-hot encoding and label encoding where necessary.
  - Normalized numerical features using `MinMaxScaler`.

## Model Architecture.
- **XGBoost Model**:
   - Optimized hyperparameters (learning rate, max depth, n_estimators, subsample, colsample_bytree).
   - Feature importance analysis to select the most relevant variables.

## Training
- **Loss function**: RMSE (Root Mean Squared Error)
- **Optimizers**: Adam (Neural Network), XGBoost default optimizer
- **Validation split**: 10% of training data
- **Number of epochs**: 50 (for the neural network)
- **Batch size**: 32

## Evaluation
The model is evaluated using:
- **RMSE**: Measures the deviation of predictions from actual sales.
- **Feature Importance**: XGBoost feature importance analysis to refine feature selection.
- **Cross-validation**: Used to ensure the model generalizes well to unseen data.

## Predictions
- Predictions are made on the test set, producing the estimated `number_of_tickets` for each ride.
- The final output is saved as `submission.csv` in the required format for submission.

## Authors
This project was developed as part of a machine learning challenge. Contributions are welcome!

## License
This project is open-source and available under the MIT License.
