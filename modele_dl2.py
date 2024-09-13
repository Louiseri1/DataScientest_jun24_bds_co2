import pandas as pd
import matplotlib.pyplot as plt
from keras import layers, models, optimizers
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump

df_original = pd.read_csv('data_2012-2015.csv', on_bad_lines="skip", sep= ',', low_memory=False)

scaler = StandardScaler()
encoder_le =  LabelEncoder()

liste_cbr = {"GO":"Gazole",
            "ES":"Essence",
            "EH":"Essence",
            "GH":"Gazole",
            "ES/GN":"Essence",
            "GN/ES":"Gaz Naturel Vehicule (GNV)",
            "ES/GP":"Essence",
            "GP/ES":"Gaz de Petrole Liquefie (GPL)",
            "EL":"Electrique",
            "GN":"Gaz Naturel Vehicule (GNV)",
            "EE":"Essence",
            "FE":"SuperEthanol-E85",
            "GL":"Gazole"}

df_original["Carburant"] = df_original["Carburant"].replace(liste_cbr)

df = df_original[["Consommation mixte (l/100km)", "Carburant", "CO2 (g/km)", "Puissance administrative","masse vide euro min (kg)"]]
df = df.dropna(how="any")

df_carb = pd.get_dummies(df["Carburant"])
df_carb = df_carb.replace({True:1, False:0})

df["Carburant"] = encoder_le.fit_transform(df["Carburant"])

X_dl = pd.concat([df[["Consommation mixte (l/100km)", "Puissance administrative","masse vide euro min (kg)"]], df_carb], axis=1)
X_dl = scaler.fit_transform(X_dl)
y_dl = df["CO2 (g/km)"]

X_dl_train, X_dl_test, y_dl_train, y_dl_test = train_test_split(X_dl, y_dl, test_size=0.2, random_state=9001)

### DEUXIÈME MODÈLE DE DEEP LEARNING
# Les 4 variables explicatives sont utilisées pour prédire CO2
# La variable carburant est encodée en différentes variables indicatrices
# 1 couche cachée avec 16 units
# X : ["Puissance administrative", "Consommation mixte (l/100km)", "masse vide euro min (kg)", "Carburant"]
# y : "CO2 (g/km)"

inputs = layers.Input((8, ), name="inputs")
dense1 = layers.Dense(16, activation="relu", name="dense1")
dense4 = layers.Dense(1, name="output")

x=dense1(inputs)
outputs=dense4(x)

optimizer = optimizers.Adam()

model_dl = models.Model(inputs = inputs, outputs = outputs)
model_dl.compile(loss="mean_squared_error", optimizer=optimizer)

# Entraînement
hist = model_dl.fit(X_dl_train, y_dl_train, epochs=100, batch_size=32, validation_split=0.1)
dump(model_dl, "model_dl2")