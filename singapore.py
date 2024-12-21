import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class SingaporeFlatPricePredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.required_features = ['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'lease_commence_date']
        
    @st.cache_data
    def load_data(_self):
        """Load and clean the dataset"""
        try:
            data = pd.read_csv(r"C:\ALL folder in dexstop\PycharmProjects\GUVI-Ai\singapoor price ai\cleaned_data\merged_resale_flats.csv", low_memory=False)
            st.success("✅ Data loaded successfully")
            
            # Drop rows with missing values in required features
            initial_len = len(data)
            data = data.dropna(subset=['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'lease_commence_date', 'resale_price'])
            st.write(f"Total records after cleaning: {len(data):,}")
            
            return data
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    @st.cache_data
    def preprocess_data(_self, data):
        """Preprocess the data for training"""
        try:
            # Create features DataFrame
            X = data[_self.required_features].copy()
            y = data['resale_price']
            
            # Calculate flat age
            X['flat_age'] = 2024 - pd.to_numeric(X['lease_commence_date'], errors='coerce')
            X = X.drop('lease_commence_date', axis=1)
            
            # Encode categorical variables and store encoders
            categorical_features = ['town', 'flat_type', 'storey_range']
            label_encoders = {}
            for feature in categorical_features:
                le = LabelEncoder()
                X[feature] = le.fit_transform(X[feature].astype(str))
                label_encoders[feature] = le
            
            st.success("✅ Data preprocessing completed")
            return X, y, label_encoders
        except Exception as e:
            st.error(f"Error preprocessing data: {str(e)}")
            return None, None, None

    @st.cache_resource
    def train_model(_self, X, y):
        """Train the Random Forest model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Initialize and train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Calculate all metrics
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_score = r2_score(y_train, train_pred)
            test_score = r2_score(y_test, test_pred)
            mae = mean_absolute_error(y_test, test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            st.success("✅ Model training completed")
            
            # Display all metrics in columns
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Score", f"{train_score:.4f}")
            with col2:
                st.metric("Testing Score", f"{test_score:.4f}")
            
            # Display additional metrics
            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric("R² Score", f"{test_score:.4f}")
            with col4:
                st.metric("MAE", f"${mae:,.2f}")
            with col5:
                st.metric("RMSE", f"${rmse:,.2f}")
            
            return model
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None
            
            # Calculate all metrics
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_score = r2_score(y_train, train_pred)
            test_score = r2_score(y_test, test_pred)
            mae = mean_absolute_error(y_test, test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            st.success("✅ Model training completed")
            
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{test_score:.4f}")
            with col2:
                st.metric("MAE", f"${mae:,.2f}")
            with col3:
                st.metric("RMSE", f"${rmse:,.2f}")
            
            return model
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None

    def predict_price(self, town, flat_type, storey_range, floor_area_sqm, lease_commence_year):
        """Predict the price of a flat"""
        try:
            # Encode inputs
            town_encoded = self.label_encoders['town'].transform([str(town)])[0]
            flat_type_encoded = self.label_encoders['flat_type'].transform([str(flat_type)])[0]
            storey_range_encoded = self.label_encoders['storey_range'].transform([str(storey_range)])[0]
            
            flat_age = 2024 - lease_commence_year
            
            # Create input data
            input_data = pd.DataFrame([[
                town_encoded,
                flat_type_encoded,
                storey_range_encoded,
                floor_area_sqm,
                flat_age
            ]], columns=['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_age'])
            
            prediction = self.model.predict(input_data)[0]
            return round(prediction, 2)
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="Singapore Flat Price Predictor",
        page_icon="🏢",
        layout="wide"
    )
    
    st.title("🏢 Singapore Flat Price Predictor")
    st.write("This app predicts HDB flat prices based on various features.")
    
    # Initialize predictor
    predictor = SingaporeFlatPricePredictor()
    
    # Load and process data (cached)
    data = predictor.load_data()
    if data is None:
        st.stop()
    
    # Preprocess data (cached)
    X, y, predictor.label_encoders = predictor.preprocess_data(data)
    if X is None or y is None:
        st.stop()
    
    # Train model (cached)
    predictor.model = predictor.train_model(X, y)
    if predictor.model is None:
        st.stop()
    
    # Create prediction interface
    st.header("📊 Predict Flat Price")
    
    col1, col2 = st.columns(2)
    
    with col1:
        town = st.selectbox("Town", sorted(data['town'].unique()))
        flat_type = st.selectbox("Flat Type", sorted(data['flat_type'].unique()))
        storey_range = st.selectbox("Storey Range", sorted(data['storey_range'].unique()))
        
    with col2:
        floor_area_sqm = st.number_input(
            "Floor Area (sqm)", 
            min_value=20.0,
            max_value=200.0,
            value=90.0,
            step=0.5
        )
        lease_commence_year = st.number_input(
            "Lease Commence Year",
            min_value=1960,
            max_value=2024,
            value=1990,
            step=1
        )
    
    # Add a predict button
    if st.button("🔮 Predict Price", type="primary"):
        prediction = predictor.predict_price(
            town=town,
            flat_type=flat_type,
            storey_range=storey_range,
            floor_area_sqm=floor_area_sqm,
            lease_commence_year=lease_commence_year
        )
        
        if prediction:
            st.success("Prediction completed!")
            st.markdown(f"### Predicted Price: :green[${prediction:,.2f}]")
            
            # Display input summary
            st.subheader("Input Summary")
            summary_data = {
                "Feature": ["Town", "Flat Type", "Storey Range", "Floor Area", "Lease Commence Year", "Flat Age"],
                "Value": [town, flat_type, storey_range, f"{floor_area_sqm} sqm", 
                         lease_commence_year, f"{2024 - lease_commence_year} years"]
            }
            st.table(pd.DataFrame(summary_data))

if __name__ == "__main__":
    main()