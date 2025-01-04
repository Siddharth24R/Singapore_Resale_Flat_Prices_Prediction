import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

class EnhancedHDBPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = RobustScaler(with_centering=True, with_scaling=True)
        self.required_columns = [
            'town', 'flat_type', 'storey_range', 
            'floor_area_sqm', 'lease_commence_date', 
            'resale_price'
        ]

    def create_location_features(self, df):
        df = df.copy()
        
        # Calculate town-level statistics
        town_stats = df.groupby('town').agg({
            'resale_price': ['mean', 'median'],
            'floor_area_sqm': 'mean'
        }).reset_index()
        
        town_stats.columns = ['town', 'town_mean_price', 'town_median_price', 'town_avg_size']
        df = df.merge(town_stats, on='town', how='left')
        
        # Price per sqm and location premium
        df['price_per_sqm'] = df['resale_price'] / df['floor_area_sqm']
        df['location_premium'] = df['price_per_sqm'] / df.groupby('town')['price_per_sqm'].transform('mean')
        
        return df

    def create_property_features(self, df):
        df = df.copy()
        
        # Storey-related features
        df[['storey_min', 'storey_max']] = df['storey_range'].str.split(' TO ', expand=True).astype(int)
        df['avg_storey'] = (df['storey_min'] + df['storey_max']) / 2
        df['high_floor'] = df['avg_storey'] > df.groupby('town')['avg_storey'].transform('median')
        
        # Lease and age features
        df['remaining_lease'] = 99 - (2024 - df['lease_commence_date'])
        df['flat_age'] = 2024 - df['lease_commence_date']
        df['lease_decay'] = (99 - df['remaining_lease']) / 99
        
        # Size categorization with handling for duplicate values
        try:
            df['size_category'] = pd.qcut(
                df['floor_area_sqm'], 
                q=5, 
                labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'],
                duplicates='drop'  # Handle duplicate bin edges
            )
        except Exception:
            # Fallback if qcut fails: use fixed bins
            size_bins = [0, 50, 75, 100, 125, float('inf')]
            df['size_category'] = pd.cut(
                df['floor_area_sqm'],
                bins=size_bins,
                labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'],
                include_lowest=True
            )
        
        # Interaction features
        df['area_lease'] = df['floor_area_sqm'] * df['remaining_lease']
        df['area_storey'] = df['floor_area_sqm'] * df['avg_storey']
        
        return df
    def handle_outliers(self, df, columns):
        df_clean = df.copy()
        
        for col in columns:
            if col in df_clean.columns:
                # IQR method
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Z-score method
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                z_scores = abs((df_clean[col] - mean) / std)
                
                # Combined filtering
                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & 
                    (df_clean[col] <= upper_bound) &
                    (z_scores <= 3)
                ]
        
        return df_clean

    def preprocess_data(self, df):
        try:
            df = df.copy()
            
            # Create features
            df = self.create_location_features(df)
            df = self.create_property_features(df)
            
            # Handle outliers for training data
            is_prediction = len(df) == 1 and df['resale_price'].iloc[0] == 0
            if not is_prediction:
                numerical_columns = ['floor_area_sqm', 'price_per_sqm']
                df = self.handle_outliers(df, numerical_columns)
            
            # Encode categorical variables
            categorical_features = ['town', 'flat_type', 'size_category']
            for col in categorical_features:
                if col in df.columns:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        self.label_encoders[col].fit(df[col].astype(str))
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
            
            # Select final features
            feature_cols = [
                'town', 'flat_type', 'avg_storey', 'floor_area_sqm',
                'remaining_lease', 'flat_age', 'lease_decay',
                'price_per_sqm', 'location_premium', 'area_lease',
                'area_storey', 'size_category', 'high_floor'
            ]
            
            # Ensure all features exist
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0
            
            X = df[feature_cols]
            
            # Scale numerical features
            numerical_features = [
                'avg_storey', 'floor_area_sqm', 'remaining_lease',
                'flat_age', 'lease_decay', 'area_lease', 'area_storey'
            ]
            
            if len(df) > 1 or not hasattr(self.scaler, 'n_features_in_'):
                self.scaler.fit(X[numerical_features])
            X[numerical_features] = self.scaler.transform(X[numerical_features])
            
            y = df['resale_price']
            
            return X, y, df, feature_cols
            
        except Exception as e:
            st.error(f"Error in preprocessing: {str(e)}")
            raise

def load_and_train():
    """Load and train the model"""
    try:
        df = pd.read_csv("C:/ALL folder in dexstop/PycharmProjects/GUVI-Ai/singapoor price ai/cleaned_data/merged_resale_flats.csv")
        
        preprocessor = EnhancedHDBPreprocessor()
        X, y, processed_df, feature_names = preprocessor.preprocess_data(df)
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=8,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Calculate metrics
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        metrics = {
            'train_score': model.score(X_train, y_train),
            'test_score': model.score(X_test, y_test),
            'mae': mean_absolute_error(y_test, test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, test_pred))
        }
        
        return model, preprocessor, df, feature_names, metrics
        
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None, None, None, None, None

def main():
    st.set_page_config(page_title="Singapore HDB Price Predictor", page_icon="🏢", layout="wide")
    
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.preprocessor = None
        st.session_state.raw_data = None
        st.session_state.feature_names = None
        st.session_state.metrics = None
    
    # Create main page tabs
    tab1, tab2, tab3 = st.tabs(["🏠 Price Prediction", "📊 Model Performance", "📈 Data Analysis"])
    
    # Load data and train model if not done
    if st.session_state.model is None:
        with st.spinner("Initializing model... Please wait."):
            model, preprocessor, df, feature_names, metrics = load_and_train()
            if model is not None:
                st.session_state.model = model
                st.session_state.preprocessor = preprocessor
                st.session_state.raw_data = df
                st.session_state.feature_names = feature_names
                st.session_state.metrics = metrics
    
    # Tab 1: Price Prediction
    with tab1:
        st.title("🏢 Singapore HDB Flat Price Predictor")
        
        # Essential inputs only
        st.sidebar.header("📝 Input Parameters")
        inputs = {}
        
        # Location details
        st.sidebar.subheader("Location Details")
        inputs['town'] = st.sidebar.selectbox(
            "Town", 
            sorted(st.session_state.raw_data['town'].unique())
        )
        
        # Property details
        st.sidebar.subheader("Property Details")
        inputs['flat_type'] = st.sidebar.selectbox(
            "Flat Type", 
            sorted(st.session_state.raw_data['flat_type'].unique())
        )
        inputs['storey_range'] = st.sidebar.selectbox(
            "Storey Range", 
            sorted(st.session_state.raw_data['storey_range'].unique())
        )
        
        # Area and lease inputs
        col1, col2 = st.sidebar.columns(2)
        with col1:
            inputs['floor_area_sqm'] = st.number_input(
                "Floor Area (sqm)",
                min_value=float(st.session_state.raw_data['floor_area_sqm'].min()),
                max_value=float(st.session_state.raw_data['floor_area_sqm'].max()),
                value=90.0,
                step=0.5
            )
        with col2:
            inputs['lease_commence_date'] = st.number_input(
                "Lease Start Year",
                min_value=int(st.session_state.raw_data['lease_commence_date'].min()),
                max_value=2024,
                value=1990,
                step=1
            )
        
        if st.sidebar.button("🔮 Predict Price", type="primary"):
            try:
                # Create DataFrame for prediction
                input_df = pd.DataFrame([inputs])
                input_df['resale_price'] = 0  # Dummy value for preprocessing
                
                # Preprocess input
                X_input, _, _, _ = st.session_state.preprocessor.preprocess_data(input_df)
                
                # Make prediction
                prediction = st.session_state.model.predict(X_input)[0]
                
                # Display prediction
                st.success("Prediction completed!")
                st.markdown(f"### Predicted Price: :green[${prediction:,.2f}]")
                
                # Display input summary and analysis
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Input Summary")
                    summary_df = pd.DataFrame({
                        "Parameter": inputs.keys(),
                        "Value": inputs.values()
                    })
                    st.table(summary_df)
                
                with col2:
                    st.subheader("Location Analysis")
                    town_avg = st.session_state.raw_data[
                        st.session_state.raw_data['town'] == inputs['town']
                    ]['resale_price'].mean()
                    
                    st.metric(
                        "Town Average Price",
                        f"${town_avg:,.2f}",
                        f"{((prediction - town_avg) / town_avg) * 100:,.1f}% difference"
                    )
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    # Tab 2: Model Performance
    with tab2:
        st.header("📊 Model Performance Metrics")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Training Score (R²)", f"{st.session_state.metrics['train_score']:.4f}")
        with col2:
            st.metric("Testing Score (R²)", f"{st.session_state.metrics['test_score']:.4f}")
        with col3:
            st.metric("Mean Absolute Error", f"${st.session_state.metrics['mae']:,.2f}")
        with col4:
            st.metric("Root Mean Squared Error", f"${st.session_state.metrics['rmse']:,.2f}")
        
        # Feature importance
        st.subheader("Feature Importance Analysis")
        importance_df = pd.DataFrame({
            'Feature': st.session_state.feature_names,
            'Importance': st.session_state.model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance in Price Prediction'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Data Analysis
    with tab3:
        st.header("📈 Market Analysis")
        
        # Price distribution by town
        st.subheader("Price Distribution by Town")
        fig = px.box(
            st.session_state.raw_data,
            x='town',
            y='resale_price',
            title='Resale Price Distribution by Town'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Price vs Floor Area
        st.subheader("Price vs Floor Area")
        fig = px.scatter(
            st.session_state.raw_data,
            x='floor_area_sqm',
            y='resale_price',
            color='flat_type',
            title='Resale Price vs Floor Area'
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
