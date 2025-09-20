import csv
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from tabulate import tabulate


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


MODEL_FILENAME = "crop_rf_model.pkl"
SCALER_FILENAME = "scaler.pkl"
TOP_N_CROPS = 5

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV file using csv.reader and convert to DataFrame."""
    logging.info(f"Loading dataset from {filepath} using csv.reader")
    with open(filepath, mode='r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        data = list(csvreader)
    df = pd.DataFrame(data, columns=header)

    logging.info(f"Dataset loaded with shape: {df.shape}")

    
    numeric_cols = ['N_SOIL', 'P_SOIL', 'K_SOIL', 'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    
    df.dropna(subset=numeric_cols, inplace=True)

    return df

def preprocess_data(df: pd.DataFrame):
    """Extract features and target, scale features."""
    features = ['N_SOIL', 'P_SOIL', 'K_SOIL', 'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL']
    X = df[features]
    y = df['CROP']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logging.info("Features scaled using StandardScaler.")
    return X_scaled, y, scaler

def train_and_tune_model(X, y):
    """Train RandomForest with hyperparameter tuning and calibration."""
    logging.info("Starting train-test split.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    logging.info("Starting GridSearchCV for hyperparameter tuning.")
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    logging.info(f"Best parameters found: {grid_search.best_params_}")

    
    calibrated_rf = CalibratedClassifierCV(best_rf, cv=5)
    calibrated_rf.fit(X_train, y_train)
    logging.info("Calibrated classifier trained.")

    
    y_pred = calibrated_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Test set accuracy: {accuracy:.4f}")

    print("\n=== Classification Report on Test Data ===")
    print(classification_report(y_test, y_pred))

    return calibrated_rf, X_train, y_train

def save_model_and_scaler(model, scaler):
    """Save model and scaler to disk."""
    joblib.dump(model, MODEL_FILENAME)
    joblib.dump(scaler, SCALER_FILENAME)
    logging.info(f"Model saved to {MODEL_FILENAME}")
    logging.info(f"Scaler saved to {SCALER_FILENAME}")

def load_model_and_scaler():
    """Load model and scaler from disk."""
    model = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
    logging.info("Model and scaler loaded from disk.")
    return model, scaler

def get_user_input():
    """Prompt user for soil and climate inputs with validation."""
    print("\n=== Crop Recommendation System ===")
    print("Please enter the following soil and climate parameters:")

    def get_float(prompt, min_val=None, max_val=None):
        while True:
            try:
                val = float(input(prompt))
                if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
                    print(f"Value must be between {min_val} and {max_val}. Please try again.")
                    continue
                return val
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    N = get_float("Nitrogen content in soil (N_SOIL) [0-140]: ", 0, 140)
    P = get_float("Phosphorus content in soil (P_SOIL) [0-145]: ", 0, 145)
    K = get_float("Potassium content in soil (K_SOIL) [0-205]: ", 0, 205)
    temperature = get_float("Temperature (°C) [-10 to 50]: ", -10, 50)
    humidity = get_float("Humidity (%) [0-100]: ", 0, 100)
    ph = get_float("Soil pH value [0-14]: ", 0, 14)
    rainfall = get_float("Rainfall (mm) [0-3000]: ", 0, 3000)

    return pd.DataFrame([{
        'N_SOIL': N,
        'P_SOIL': P,
        'K_SOIL': K,
        'TEMPERATURE': temperature,
        'HUMIDITY': humidity,
        'ph': ph,
        'RAINFALL': rainfall
    }])

def get_user_budget():
    """Prompt user for budget input with validation."""
    while True:
        try:
            budget_input = input("\nEnter your budget for crop cultivation (in your currency): ").strip()
            budget = float(budget_input)
            if budget <= 0:
                print("Error: Budget must be a positive number. Please try again.")
                continue
            return budget
        except ValueError:
            print("Error: Invalid input. Please enter a numeric value for the budget.")

def get_crop_costs():
    """Return a dictionary of crop costs. Update as needed."""
    return {
        'rice': 5000,
        'wheat': 3000,
        'maize': 4000,
        'cotton': 7000,
        'sugarcane': 6000,
        'barley': 3500,
        'millet': 2500,
        'oilseed': 4500,
        'pulses': 3200,
        'groundnut': 4800,
        'coffee': 8000,
        'chickpea': 14000,
        
    }

def suggest_top_n_crops(model, scaler, sample, budget, crop_costs, top_n=5):
    """Predict and return top N crops with probabilities, costs, and budget fit."""
    sample_scaled = scaler.transform(sample)
    probabilities = model.predict_proba(sample_scaled)[0]
    classes = model.classes_

    crop_info = []
    for crop, prob in zip(classes, probabilities):
        cost = crop_costs.get(crop.lower())
        if cost is None:
            logging.warning(f"Cost data missing for crop '{crop}'. Skipping.")
            continue
        within_budget = cost <= budget
        crop_info.append((crop, prob, cost, within_budget))

    crop_info.sort(key=lambda x: x[1], reverse=True)
    return crop_info[:top_n]

def display_top_crops(top_crops):
    """Display the top crop recommendations in a formatted table."""
    if not top_crops:
        print("\nNo crop suggestions found with available cost data.")
        return

    headers = ["Rank", "Crop", "Probability (%)", "Estimated Cost", "Within Budget"]
    table = []
    for idx, (crop, prob, cost, within_budget) in enumerate(top_crops, start=1):
        prob_percent = prob * 100
        within_budget_str = "Yes" if within_budget else "No"
        table.append([idx, crop.title(), f"{prob_percent:.2f}", f"{cost:,}", within_budget_str])

    print("\n=== Top Crop Recommendations Based on Your Inputs ===\n")
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

def generate_yield_tips(sample: pd.DataFrame, budget: float, crop: str):
    tips = []
    N = sample.at[0, 'N_SOIL']
    P = sample.at[0, 'P_SOIL']
    K = sample.at[0, 'K_SOIL']
    temperature = sample.at[0, 'TEMPERATURE']
    humidity = sample.at[0, 'HUMIDITY']
    ph = sample.at[0, 'ph']
    rainfall = sample.at[0, 'RAINFALL']

    tips.append("1. Use certified high-yield seed varieties suitable for your crop.")
    tips.append("2. Follow recommended planting dates to match the crop's growth cycle.")

    if N < 50:
        tips.append("3. Apply nitrogen-rich fertilizers (e.g., urea) to improve soil fertility.")
    else:
        tips.append("3. Maintain current nitrogen levels; avoid over-fertilization to save costs.")

    if P < 30:
        tips.append("4. Add phosphorus fertilizers (e.g., single super phosphate) to enhance root development.")
    else:
        tips.append("4. Phosphorus levels are adequate; focus on balanced fertilization.")

    if K < 40:
        tips.append("5. Use potassium fertilizers (e.g., muriate of potash) to improve drought resistance.")
    else:
        tips.append("5. Potassium levels are sufficient; monitor soil regularly.")

    if ph < 6.0:
        tips.append("6. Apply agricultural lime to raise soil pH for better nutrient availability.")
    elif ph > 7.5:
        tips.append("6. Use sulfur or acidifying fertilizers to lower soil pH if needed.")
    else:
        tips.append("6. Soil pH is optimal; maintain current practices.")

    if temperature < 15:
        tips.append("7. Use mulching to retain soil warmth and moisture.")
    elif temperature > 35:
        tips.append("7. Implement shading nets or irrigation scheduling to reduce heat stress.")

    if humidity < 40:
        tips.append("8. Use drip irrigation to conserve water and maintain soil moisture.")
    elif humidity > 80:
        tips.append("8. Ensure proper drainage to prevent fungal diseases.")

    if rainfall < 500:
        tips.append("9. Consider supplemental irrigation to meet crop water needs.")
    elif rainfall > 1500:
        tips.append("9. Use raised beds or proper drainage to avoid waterlogging.")

    if budget < 3000:
        tips.append("10. Prioritize low-cost, high-impact practices like proper spacing and timely weeding.")
    elif budget < 7000:
        tips.append("10. Invest in quality fertilizers and pest management for better yield.")
    else:
        tips.append("10. Consider advanced techniques like soil testing, mechanized planting, and precision farming.")

    crop_lower = crop.lower()
    if crop_lower == 'rice':
        tips.append("11. Use System of Rice Intensification (SRI) methods to increase yield with less water.")
    elif crop_lower == 'wheat':
        tips.append("11. Practice crop rotation and timely irrigation to improve wheat yield.")
    elif crop_lower == 'maize':
        tips.append("11. Use proper spacing and timely fertilizer application for maize.")
    

    return tips

def check_agriculture_feasibility(sample: pd.DataFrame) -> bool:
    N = sample.at[0, 'N_SOIL']
    P = sample.at[0, 'P_SOIL']
    K = sample.at[0, 'K_SOIL']
    temperature = sample.at[0, 'TEMPERATURE']
    humidity = sample.at[0, 'HUMIDITY']
    ph = sample.at[0, 'ph']
    rainfall = sample.at[0, 'RAINFALL']

    if N < 5:
        print("Nitrogen level too low for most crops.")
        return False
    if P < 5:
        print("Phosphorus level too low for most crops.")
        return False
    if K < 5:
        print("Potassium level too low for most crops.")
        return False
    if temperature < 5 or temperature > 45:
        print("Temperature outside suitable range for most crops (5°C - 45°C).")
        return False
    if humidity < 20 or humidity > 90:
        print("Humidity outside suitable range for most crops (20% - 90%).")
        return False
    if ph < 4.5 or ph > 9.0:
        print("Soil pH outside suitable range for most crops (4.5 - 9.0).")
        return False
    if rainfall < 100 or rainfall > 2500:
        print("Rainfall outside suitable range for most crops (100mm - 2500mm).")
        return False

    return True

def main():
    filename = "indiancrop_dataset.csv"  

    try:
        df = load_data(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    X_scaled, y, scaler = preprocess_data(df)

    
    try:
        model, scaler = load_model_and_scaler()
        logging.info("Loaded existing model and scaler.")
    except (FileNotFoundError, IOError):
        logging.info("No saved model found. Training new model...")
        model, X_train, y_train = train_and_tune_model(X_scaled, y)
        save_model_and_scaler(model, scaler)

    sample = get_user_input()

    if not check_agriculture_feasibility(sample):
        print("\nInput conditions not suitable for agriculture. Adjust inputs or consult experts.")
        return

    budget = get_user_budget()
    crop_costs = get_crop_costs()

    top_crops = suggest_top_n_crops(model, scaler, sample, budget, crop_costs, top_n=TOP_N_CROPS)

    if top_crops:
        predicted_crop = top_crops[0][0]
        print(f"\nMost suitable crop based on your inputs: >>> {predicted_crop.upper()} <<<")
    else:
        print("\nCould not determine a suitable crop based on inputs and cost data.")
        return

    display_top_crops(top_crops)

    tips = generate_yield_tips(sample, budget, predicted_crop)
    print("\n=== Practical Tips to Increase Your Crop Yield (up to 200%) ===\n")
    for tip in tips:
        print(tip)

if __name__ == "__main__":
    main()
