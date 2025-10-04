# Importações continuam as mesmas...
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

def run_classification(dataset, dataset_name):
    print(f"--- Iniciando Classificação para a Base de Dados: {dataset_name} ---")

    # 2. Carregamento e Preparação dos Dados
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target)
    class_names = dataset.target_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Escalonamento dos Dados
    print("Aplicando escalonamento nos dados (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Desenvolvimento e Treinamento do Modelo MLPClassifier (VERSÃO CORRIGIDA)
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=1000,
        random_state=42,
        solver='adam',
        activation='relu'
        # A linha 'early_stopping=True' foi removida.
    )
    
    mlp.fit(X_train_scaled, y_train)

    # 4. Realização das Predições
    y_pred = mlp.predict(X_test_scaled)

    # 5. Avaliação do Classificador
    print("\nMétricas de Avaliação:")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão (Ponderada): {precision:.4f}")
    print(f"Revocação (Ponderada): {recall:.4f}\n")
    print("Relatório de Classificação Detalhado:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 6. Geração e Plotagem da Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title(f'Matriz de Confusão - {dataset_name}')
    plt.show()
    print("-" * 50 + "\n")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall}

# --- Execução ---
iris_data = load_iris()
wine_data = load_wine()

results_mlp_iris = run_classification(iris_data, "Iris")
results_mlp_wine = run_classification(wine_data, "Wine")