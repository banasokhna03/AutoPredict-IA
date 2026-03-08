from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# --- ENTRAÎNEMENT DE L'IA AU DÉMARRAGE ---
df = pd.read_csv('car data.csv')
df.columns = df.columns.str.strip()
COEFF_CFA = 750000 
df['Selling_Price_CFA'] = df['Selling_Price'] * COEFF_CFA
df['Present_Price_CFA'] = df['Present_Price'] * COEFF_CFA
df['Age'] = 2026 - df['Year']

df_ml = pd.get_dummies(df, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)
features = ['Present_Price_CFA', 'Driven_kms', 'Age', 'Owner'] + \
           [col for col in df_ml.columns if any(x in col for x in ['Fuel_', 'Selling_type_', 'Transmission_'])]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(df_ml[features], df_ml['Selling_Price_CFA'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    p_neuf = float(data['price'])
    kms = int(data['kms'])
    
    results = {}
    for label, modif in [("passe", -3), ("present", 0), ("futur", 3)]:
        input_row = pd.DataFrame(0, index=[0], columns=features)
        input_row['Present_Price_CFA'] = p_neuf
        input_row['Age'] = 5 + modif
        input_row['Driven_kms'] = kms + (30000 if modif > 0 else 0)
        
        prediction = model.predict(input_row)[0]
        results[label] = f"{int(prediction):,}".replace(",", " ") + " FCFA"
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)