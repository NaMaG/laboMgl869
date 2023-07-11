import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Charger les données à partir du fichier CSV
print("Chargement des données...")
data = pd.read_csv('averagesWithBugs.csv')

# Certaines colonnes peuvent contenir des valeurs nulles qui pourraient poser problème lors de l'entraînement du modèle.
# Nous les supprimons ici pour simplifier.
data = data.dropna()

# Convertir les colonnes catégorielles en numériques
le = LabelEncoder()
data['Version'] = le.fit_transform(data['Version'])
data['Files'] = le.fit_transform(data['Files'])

# Diviser les données en variables indépendantes (X) et variable dépendante (y)
print("Préparation des données...")
X = data.drop('ContainsBug', axis=1)
y = data['ContainsBug']

# Diviser les données en ensembles d'entraînement et de test
print("Création des ensembles d'entraînement et de test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instancier le modèle de régression logistique
print("Création du modèle de régression logistique...")
model_lr = LogisticRegression(max_iter=50000, solver = 'lbfgs', random_state = 42, warm_start = True)
model_rf = RandomForestClassifier(max_depth = 10, random_state = 42, warm_start = True)

# Entraîner le modèle sur les données d'entraînement
print("Entraînement du modèle...")
model_lr.fit(X_train, y_train)
model_rf.fit(X_train, y_train)

# Faire des prédictions sur les données de test
print("Prédictions à l'aide du modèle entraîné...")
y_pred_lr = model_lr.predict(X_test)
y_pred_rf = model_rf.predict(X_test)


# Calculer les métriques de performance
cnf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
cnf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

performance_lr = (cnf_matrix_lr[0, 0] + cnf_matrix_lr[1, 1]) / cnf_matrix_lr.sum()
performance_rf = (cnf_matrix_rf[0, 0] + cnf_matrix_rf[1, 1]) / cnf_matrix_rf.sum()

print(f'Performance du modèle de régression logistique : {performance_lr}')
print(f'Performance du modèle de forêt aléatoire : {performance_rf}')

# Calculer les métriques de performance
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
accuracy_lr = np.mean(y_pred_lr == y_test)
auc_lr = roc_auc_score(y_test, y_pred_lr)

precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
accuracy_rf = np.mean(y_pred_rf == y_test)
auc_rf = roc_auc_score(y_test, y_pred_rf)


print(f'Logistic Regression - Précision : {precision_lr}\nRappel : {recall_lr}\nJustesse : {accuracy_lr}\nAUC : {auc_lr}')
print(f'Random Forest - Précision : {precision_rf}\nRappel : {recall_rf}\nJustesse : {accuracy_rf}\nAUC : {auc_rf}')

# Calculer l'importance des variables
result = permutation_importance(model_lr, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx])
ax.set_title("Importance des variables selon le Permutation Importance")
plt.savefig('monogramme.png')

# Save contructed models for later use
joblib.dump(model_lr, 'lr_model.dump')
joblib.dump(model_rf, 'rf_model.dump')





