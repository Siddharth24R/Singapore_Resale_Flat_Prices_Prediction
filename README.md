# Singapore Flat Price Predictor 🏢

This Streamlit-based web application predicts the resale prices of HDB flats in Singapore using machine learning. It leverages historical data, preprocessed features, and a trained Random Forest Regressor to provide accurate price estimates.

## Features

- **Dynamic User Interface**: Interactive interface for users to input flat features like town, flat type, storey range, floor area, and lease commence year.
- **Data Preprocessing**: Automatic cleaning and encoding of data to handle categorical and numerical variables.
- **Machine Learning Model**: Predicts flat prices using a Random Forest Regressor.
- **Evaluation Metrics**: Displays training and testing scores, along with metrics like MAE and RMSE.
- **Cached Performance**: Optimized with Streamlit's caching to improve performance.

## Installation

1. Clone the repository or download the code.
2. Install the required Python libraries:
   ```bash
   pip install pandas numpy scikit-learn streamlit
   ```
3. Place the cleaned dataset (`merged_resale_flats.csv`) in the `cleaned_data` directory:
   ```
   C:\ALL folder in dexstop\PycharmProjects\GUVI-Ai\singapoor price ai\cleaned_data\merged_resale_flats.csv
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```
2. Open the app in your browser at `http://localhost:8501`.
3. Input the flat details in the provided fields and click on **Predict Price** to see the estimated resale value.

## Dataset

The dataset (`merged_resale_flats.csv`) is preprocessed and cleaned, including:

- **Required Features**: `town`, `flat_type`, `storey_range`, `floor_area_sqm`, `lease_commence_date`, and `resale_price`.
- **Derived Features**: `flat_age` (calculated as `2024 - lease_commence_date`).

## Model

- **Algorithm**: Random Forest Regressor
- **Parameters**:
  - `n_estimators=100`
  - `max_depth=20`
  - `min_samples_split=10`
  - `min_samples_leaf=5`
  - `random_state=42`

## Key Metrics

- **Training Score (R²)**: Measures the model's performance on the training data.
- **Testing Score (R²)**: Measures the model's generalization to unseen data.
- **MAE (Mean Absolute Error)**: Average magnitude of prediction errors.
- **RMSE (Root Mean Squared Error)**: Standard deviation of prediction errors.

## Sample Input

| Feature          | Example Input |
|-------------------|---------------|
| Town             | Jurong West   |
| Flat Type        | 3-Room        |
| Storey Range     | 10 to 12      |
| Floor Area (sqm) | 90.5          |
| Lease Year       | 1995          |

## Sample Output

| Feature         | Value           |
|------------------|-----------------|
| Predicted Price | $480,000.00     |

