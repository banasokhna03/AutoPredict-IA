from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# --- CONFIGURATION IABD ---
COEFF_CFA = 750000 

# Chargement et nettoyage
df = pd.read_csv('car data.csv')
df.columns = df.columns.str.strip()

# Préparation des prix et de l'âge
df['Selling_Price_CFA'] = df['Selling_Price'] * COEFF_CFA
df['Present_Price_CFA'] = df['Present_Price'] * COEFF_CFA
df['Age'] = 2026 - df['Year']

# Encodage des variables catégorielles
df_ml = pd.get_dummies(df, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)

# Définition des colonnes d'entraînement
features = ['Present_Price_CFA', 'Driven_kms', 'Age', 'Owner'] + \
           [col for col in df_ml.columns if 'Fuel_Type_' in col]

# On entraîne le modèle
X = df_ml[features]
y = df_ml['Selling_Price_CFA']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    car_name = data.get('name', 'Véhicule')
    year_val = int(data['year'])
    kms_val = int(data['kms'])
    fuel_val = data['fuel'] # Petrol ou Diesel

    # On définit un prix moyen de référence (ex: 20M FCFA) pour la base de calcul
    prix_ref = 20000000 
    
    predictions = {}
    for label, modif in [("passe", -3), ("present", 0), ("futur", 3)]:
        # Création du vecteur de données pour l'IA
        row = pd.DataFrame(0, index=[0], columns=features)
        row['Present_Price_CFA'] = prix_ref
        row['Age'] = (2026 - year_val) + modif
        row['Driven_kms'] = kms_val + (25000 if modif > 0 else 0)
        row['Owner'] = 0
        
        # Activation de la colonne carburant
        fuel_col = f'Fuel_Type_{fuel_val}'
        if fuel_col in features:
            row[fuel_col] = 1
            
        res = model.predict(row)[0]
        predictions[label] = f"{int(res):,}".replace(",", " ") + " FCFA"
    
    return jsonify({'name': car_name, 'results': predictions})

if __name__ == '__main__':
    app.run(debug=True)