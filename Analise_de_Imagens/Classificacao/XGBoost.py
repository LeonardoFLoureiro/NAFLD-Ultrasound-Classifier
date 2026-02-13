# --------------------------------------------------------------------------------
# Arquivo: C:\Users\Leonardo Loureiro\OneDrive\Documentos\Puc\PAI\G10\PAI_Trabalho_2024\Analise_de_Imagens\Classificacao\XGBoost.py
# --------------------------------------------------------------------------------

# Importar bibliotecas
import ast
import csv
from tkinter import Image, messagebox
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
import seaborn as sns


def classificar_xgboost():
    # Caminho para o arquivo CSV
    caracteristicas_csv_path = os.path.join("PAI_Trabalho_2024", "Database", "rois_caracteristicas.csv")

    dados = []
    rotulos = []


    with open(caracteristicas_csv_path, mode='r', newline='', encoding='utf-8') as arquivo:
        leitor_csv = csv.reader(arquivo)

        for linha in leitor_csv:
            
            classe = linha[1]
            haralick_entropia = ast.literal_eval(linha[3]) 
            haralick_homogeneidade = ast.literal_eval(linha[4])
            tamura = ast.literal_eval(linha[5])

            registro  = {
            'haralick_entropia_1': haralick_entropia[1],
            'haralick_entropia_2': haralick_entropia[2],
            'haralick_entropia_4': haralick_entropia[4],
            'haralick_entropia_8': haralick_entropia[8],
            'haralick_homogeneidade_1': haralick_homogeneidade[1],
            'haralick_homogeneidade_2': haralick_homogeneidade[2],
            'haralick_homogeneidade_4': haralick_homogeneidade[4],
            'haralick_homogeneidade_8': haralick_homogeneidade[8],
            'tamura_coarseness': tamura['coarseness'],
            'tamura_contrast': tamura['contrast'],
            'tamura_directionality': tamura['directionality'],
            'tamura_line_likeness': tamura['line_likeness'],
            'tamura_regularity': tamura['regularity'],
            'tamura_roughness': tamura['roughness'],

            }
            dados.append(registro)
            rotulos.append(classe) 

    # Converter para DataFrame
    X = pd.DataFrame(dados)
    label_encoder = LabelEncoder()
    tmp = pd.Series(rotulos)
    Y = label_encoder.fit_transform(tmp)


    #print("Distribuição das classes:", np.bincount(Y))

    # Criar o modelo XGBoost
    model = XGBClassifier(eval_metric='mlogloss')

    # Realizar validação cruzada (55 folds)
    scores = cross_validate(
        model, 
        X, Y, 
        cv=55, 
        scoring=['accuracy', 'precision', 'recall'],  # Métricas: Acurácia, Precisão, Recall
        return_train_score=False
    )

    # Mostrar resultados
    acuracia = scores['test_accuracy'].mean()
    sensibilidade = scores['test_recall'].mean()



    # Obter as previsões de validação cruzada
    y_pred = cross_val_predict(model, X, Y, cv=55)

    # Calcular a matriz de confusão
    cm = confusion_matrix(Y, y_pred)

    # Calcular a Especificidade
    tn, fp, fn, tp = cm.ravel()
    especificidade = tn / (tn + fp)

    # Plotar a matriz
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y), yticklabels=np.unique(Y))

    plt.title("Matriz de Confusão")
    plt.xlabel("Previsões")
    plt.ylabel("Verdadeiros")

    # Exibir o gráfico
    plt.show()
    messagebox.showinfo("Metricas", f"Accuracy Média: {acuracia:.2f}\n Especificidade Média:{especificidade:.2f}\nSensibilidade Média:{sensibilidade:.2f}\n  ")
    