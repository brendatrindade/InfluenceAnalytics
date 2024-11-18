# Importações necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Função para converter valores abreviados em números
def convert_to_number(value):
    if isinstance(value, str):
        value = value.lower().replace(',', '').replace(' ', '')
        if 'k' in value:
            return float(value.replace('k', '')) * 1e3
        elif 'm' in value:
            return float(value.replace('m', '')) * 1e6
        elif 'b' in value:
            return float(value.replace('b', '')) * 1e9
        elif '%' in value:
            return float(value.replace('%', '')) / 100
        else:
            return float(value)
    return value

# Carregar os dados
data = pd.read_csv("top_insta_influencers_data.csv")

# Converter colunas numéricas abreviadas
columns_to_convert = ['posts', 'followers', 'avg_likes', '60_day_eng_rate', 
                      'new_post_avg_like', 'total_likes']
for col in columns_to_convert:
    data[col] = data[col].apply(convert_to_number)

# Lidando com valores ausentes
data = data.dropna(subset=['60_day_eng_rate'])  # Remover linhas sem a variável dependente
data = data.fillna(data.mean())  # Preencher valores ausentes restantes com a média

# Seleção de variáveis
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Matriz de Correlação")
plt.show()

# Variáveis mais correlacionadas com a taxa de engajamento
target = "60_day_eng_rate"
features = ['followers', 'avg_likes', 'new_post_avg_like', 'influence_score']

# Dividir dados em conjuntos de treinamento e teste
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinar modelo de regressão linear
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Prever e avaliar desempenho
y_pred = linear_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Desempenho do Modelo de Regressão Linear:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# Regularização com Lasso e Ridge
for model_name, model in [("Lasso", Lasso(alpha=0.01)), ("Ridge", Ridge(alpha=0.01))]:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# Validação cruzada
cv_scores = cross_val_score(linear_model, X_train_scaled, y_train, cv=5, scoring="r2")
print("Validação Cruzada (R²):", cv_scores)
print("R² médio:", np.mean(cv_scores))

# Visualizar resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.xlabel("Valores Reais")
plt.ylabel("Valores Previstos")
plt.title("Regressão Linear: Valores Reais vs Previstos")
plt.show()
