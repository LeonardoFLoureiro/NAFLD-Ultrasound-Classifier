# --------------------------------------------------------------------------------
# Arquivo: C:\Users\Leonardo Loureiro\OneDrive\Documentos\Puc\PAI\G10\PAI_Trabalho_2024\Analise_de_Imagens\Classificacao\MobileNet.py
# --------------------------------------------------------------------------------

import ast
import csv
import pickle
from tkinter import messagebox
from matplotlib import pyplot as plt
import numpy as np
from sklearn.calibration import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import preprocess_input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model



def carregar_base_de_dados():
    # Caminho para os dados
    caracteristicas_csv_path = os.path.join("PAI_Trabalho_2024", "Database", "rois_caracteristicas.csv")
    image_size = (224, 224)
    dados = []
    rotulos = []
    pacientes = []
    with open(caracteristicas_csv_path, mode='r', newline='', encoding='utf-8') as arquivo:
        leitor_csv = csv.reader(arquivo)

        for linha in leitor_csv:
            
            path_arquivo = linha[0]
            classe = linha[1]
            partes = path_arquivo.split('_')
            paciente_id = partes[1]
            path_arquivo = os.path.join("PAI_Trabalho_2024", "Database", "ROIS_figado", path_arquivo)
            # Carregar imagem e redimensionar
            image = tf.io.read_file(path_arquivo)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, image_size)

            # Pré-processar para MobileNet
            image = preprocess_input(image)
            
            dados.append(image)
            rotulos.append(classe)
            pacientes.append(paciente_id)

    dados = np.array(dados)
    rotulos = np.array(rotulos)
    pacientes = np.array(pacientes)

    return dados,rotulos,pacientes

def carregar_modelo_nao_treinado():

    print("GPUs:", tf.config.list_physical_devices('GPU'))
    # Hiperparâmetros
    image_size = (224, 224)
    batch_size = 32
    epochs = 10

    # Carregar base
    dados,rotulos,pacientes = carregar_base_de_dados()
    label_encoder = LabelEncoder()
    rotulos = label_encoder.fit_transform(rotulos)
    #Cross Validation
    logo = LeaveOneGroupOut()
    accuracy_per_fold = []
    conf_matrices = []
    resultados_por_fold = []

    for fold_index,(train, test) in enumerate(logo.split(dados, rotulos, groups=pacientes)):

        # Divisão de treino e teste
        x_train, x_test = dados[train], dados[test]
        y_train, y_test = rotulos[train], rotulos[test]

        # Criar modelo MobileNet
        base_model = MobileNet(input_shape=(image_size[0], image_size[1], 3), include_top=False, weights='imagenet')
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(len(np.unique(rotulos)), activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        
        for layer in base_model.layers:
            layer.trainable = False

        # Compilar o modelo
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Treinar o modelo
        resultados = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,validation_data=(x_test, y_test))

        # Salvar o modelo
        model_save_path = os.path.join("PAI_Trabalho_2024", "Database","Modelo_Mobilenet", f"modelo_fold_{fold_index}.h5")
        model.save(model_save_path)

        # Teste
        y_pred = np.argmax(model.predict(x_test), axis=1) 
        scores = model.evaluate(x_test, y_test, verbose=0)
        print(f"Accuracy: {scores[1] * 100:.2f}%")
        accuracy_per_fold.append(scores[1] * 100)

        # Gerar matriz de confusão
        conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(rotulos))



        conf_matrices.append(conf_matrix)
        resultados_por_fold.append(resultados.history)

        # Salvar o histórico em um arquivo
        history_file_path = os.path.join("PAI_Trabalho_2024", "Database","Modelo_Mobilenet","historico.pkl")
        with open(history_file_path, 'wb') as f:
            pickle.dump(resultados.history, f)


    # Resultados finais
    for i,conf_matrix in enumerate(conf_matrices):
        matriz = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
        matriz.plot(cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusão - Fold {i + 1}")
        plt.show()

    # Plotar matriz de confusão agregada
    aggregated_matrix = np.sum(conf_matrices, axis=0)
    disp = ConfusionMatrixDisplay(confusion_matrix=aggregated_matrix, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão Agregada")
    plt.show()

    # Acuracias medias
    train_accuracies = []
    val_accuracies = []

    for history in resultados_por_fold:
        train_accuracies.append(history['accuracy'])
        val_accuracies.append(history['val_accuracy'])

    train_accuracies = np.array(train_accuracies)
    val_accuracies = np.array(val_accuracies)

    mean_train_accuracy = np.mean(train_accuracies, axis=0)
    mean_val_accuracy = np.mean(val_accuracies, axis=0)
    #Plotar
    plt.figure(figsize=(12, 6))
    plt.plot(mean_train_accuracy, label='Média de Treinamento', color='b')
    plt.plot(mean_val_accuracy, label='Média de Validação', color='g', linestyle='--')

    plt.title('Gráficos de Aprendizado Médios (Acurácia de Treino e Teste)')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)
    plt.show()
    messagebox.showinfo("Metricas", f"Accuracy Média: {np.mean(accuracy_per_fold):.2f}%")

def carregar_modelo_treinado():

    print("GPUs:", tf.config.list_physical_devices('GPU'))
    # Hiperparâmetros
    image_size = (224, 224)
    batch_size = 32
    epochs = 10

    # Carregar base
    dados,rotulos,pacientes = carregar_base_de_dados()
    label_encoder = LabelEncoder()
    rotulos = label_encoder.fit_transform(rotulos)
    #Cross Validation
    logo = LeaveOneGroupOut()
    accuracy_per_fold = []
    conf_matrices = []
    resultados_por_fold = []
    for fold_index,(train, test) in enumerate(logo.split(dados, rotulos, groups=pacientes)):

        # Divisão de treino e teste
        x_train, x_test = dados[train], dados[test]
        y_train, y_test = rotulos[train], rotulos[test]

        # Carregar o modelo
        model_path = os.path.join("PAI_Trabalho_2024", "Database","Modelo_Mobilenet", f"modelo_fold_{fold_index}.h5")
        model = load_model(model_path)

        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        
        # Teste
        y_pred = np.argmax(model.predict(x_test), axis=1) 
        scores = model.evaluate(x_test, y_test, verbose=0)
        print(f"Accuracy: {scores[1] * 100:.2f}%")
        accuracy_per_fold.append(scores[1] * 100)

        # Gerar matriz de confusão
        conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(rotulos))



        conf_matrices.append(conf_matrix)
        # Carregar o histórico de um arquivo
        history_file_path = history_file_path = os.path.join("PAI_Trabalho_2024", "Database","Modelo_Mobilenet","historico.pkl")
        with open(history_file_path, 'rb') as f:
            history_loaded = pickle.load(f)

        resultados_por_fold.append(history_loaded)


    # Resultados finais
    for i,conf_matrix in enumerate(conf_matrices):
        matriz = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
        matriz.plot(cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusão - Fold {i + 1}")
        plt.show()
  

    # Plotar matriz de confusão agregada
    aggregated_matrix = np.sum(conf_matrices, axis=0)
    disp = ConfusionMatrixDisplay(confusion_matrix=aggregated_matrix, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão Agregada")
    plt.show()

    # Acuracias medias
    train_accuracies = []
    val_accuracies = []

    for history in resultados_por_fold:
        train_accuracies.append(history['accuracy'])
        val_accuracies.append(history['val_accuracy'])

    train_accuracies = np.array(train_accuracies)
    val_accuracies = np.array(val_accuracies)

    mean_train_accuracy = np.mean(train_accuracies, axis=0)
    mean_val_accuracy = np.mean(val_accuracies, axis=0)
    #Plotar
    plt.figure(figsize=(12, 6))
    plt.plot(mean_train_accuracy, label='Média de Treinamento', color='b')
    plt.plot(mean_val_accuracy, label='Média de Validação', color='g', linestyle='--')

    plt.title('Gráficos de Aprendizado Médios (Acurácia de Treino e Teste)')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)
    plt.show()

    messagebox.showinfo("Metricas", f"Accuracy Média: {np.mean(accuracy_per_fold):.2f}%")