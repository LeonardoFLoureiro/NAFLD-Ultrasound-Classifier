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
    


# --------------------------------------------------------------------------------
# Arquivo: C:\Users\Leonardo Loureiro\OneDrive\Documentos\Puc\PAI\G10\PAI_Trabalho_2024\Analise_de_Imagens\Descrição\GLCM.py
# --------------------------------------------------------------------------------

from skimage.feature import graycomatrix, graycoprops
from tkinter import filedialog, messagebox
import numpy as np

def calc_glcm(roi):
    array = np.array(roi.convert("L"))

    distances = [1, 2, 4, 8] 

    results_glcm = {}
    glcms = {}

    props_glcm = ['ASM', 'homogeneity', 'dissimilarity', 'correlation', 'energy', 'contrast']

    for d in distances:
        glcm = graycomatrix(array, distances=[d], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)

        glcms[d] = glcm

        results_glcm[d] = {prop: graycoprops(glcm, prop).mean() for prop in props_glcm}



    return results_glcm, glcms

# --------------------------------------------------------------------------------
# Arquivo: C:\Users\Leonardo Loureiro\OneDrive\Documentos\Puc\PAI\G10\PAI_Trabalho_2024\Analise_de_Imagens\Descrição\Haralick.py
# --------------------------------------------------------------------------------

import numpy as np
from skimage.feature import graycomatrix, graycoprops

def calc_haralick_entropia(glcm):
    glcm_normalizada = glcm[:, :, 0, 0]
    return -np.sum(glcm_normalizada * np.log2(glcm_normalizada + 1e-10))

def calc_haralick_homogeneidade(glcm):
    return graycoprops(glcm, 'homogeneity')[0, 0]


# --------------------------------------------------------------------------------
# Arquivo: C:\Users\Leonardo Loureiro\OneDrive\Documentos\Puc\PAI\G10\PAI_Trabalho_2024\Analise_de_Imagens\Descrição\Tamura.py
# --------------------------------------------------------------------------------

# NT = 3
#       matriculaDaniel = 708726
#       matriculaGabriel = 719316
#       matriculaLeonardo = 727225

import numpy as np

def calc_tamura(roi):
    roi_array = np.array(roi)
    coarseness = _calcular_eixo_coarseness(roi_array, 0) + _calcular_eixo_coarseness(roi_array, 1)
    contrast = np.var(roi_array)
    directionality = np.mean(np.angle(np.gradient(roi_array)[0] + 1j * np.gradient(roi_array)[1]))
    line_likeness = np.mean(np.abs(np.diff(roi_array)))  
    regularity = np.std(roi_array)
    roughness = np.mean(np.abs(np.gradient(roi_array)))

    return {
        "coarseness": coarseness,
        "contrast": contrast,
        "directionality": directionality,
        "line_likeness": line_likeness,
        "regularity": regularity,
        "roughness": roughness
    }

def _calcular_eixo_coarseness(roi, axis):
    return np.mean(np.abs(np.diff(roi, axis)))

# --------------------------------------------------------------------------------
# Arquivo: C:\Users\Leonardo Loureiro\OneDrive\Documentos\Puc\PAI\G10\PAI_Trabalho_2024\Database\CRUD_csv.py
# --------------------------------------------------------------------------------

import csv

def create_coordenadas(nome_arquivo,xSUP_figado,ySUP_figado,xSUP_rim,ySUP_rim,HI,paciente):
    classe = ""
    if paciente <= 16:
        classe = "Saudavel"
    else:
        classe = "Esteatose"

    with open("PAI_Trabalho_2024/Database/rois_coordenadas.csv", mode='a', newline='') as arquivo_csv:
        escritor = csv.writer(arquivo_csv)
        escritor.writerow([nome_arquivo,classe,xSUP_figado,ySUP_figado,xSUP_rim,ySUP_rim,HI])

def create_caracteristicas(nome_arquivo,paciente,glcms_results,haralick_entropia,haralick_homogeneidade,tamura):
    classe = ""
    if paciente <= 16:
        classe = "Saudavel"
    else:
        classe = "Esteatose"

    with open("PAI_Trabalho_2024/Database/rois_caracteristicas.csv", mode='a', newline='') as arquivo_csv:
        escritor = csv.writer(arquivo_csv)
        escritor.writerow([nome_arquivo,classe,glcms_results,haralick_entropia,haralick_homogeneidade,tamura])


# --------------------------------------------------------------------------------
# Arquivo: C:\Users\Leonardo Loureiro\OneDrive\Documentos\Puc\PAI\G10\PAI_Trabalho_2024\Screens\Base.py
# --------------------------------------------------------------------------------

import tkinter as tk

class BaseScreen(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #-----------------------------
        #       Configurações
        #-----------------------------

        self.title("Generico")
        self.geometry("600x600")
        self.configure(bg='gray')

        #---------------------------
        #        Componentes
        #---------------------------



# --------------------------------------------------------------------------------
# Arquivo: C:\Users\Leonardo Loureiro\OneDrive\Documentos\Puc\PAI\G10\PAI_Trabalho_2024\Screens\Classificacao.py
# --------------------------------------------------------------------------------

#NC = 1
#ND = 2
import tkinter as tk


class ClassificacaoScreen(BaseScreen):
    def __init__(self,manager, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #-----------------------------
        #       Configurações
        #-----------------------------

        self.title("Classifiicação")
        self.geometry("600x400")
        self.configure(bg='gray')

        #---------------------------
        #        Componentes
        #---------------------------
        # Menu principal

        self.frame_entrada = tk.Frame(self)
        self.frame_entrada.pack(side=tk.TOP, pady=100, fill=tk.X)
        
            # Menu para selecionar o classificador
        self.classificador = tk.StringVar(self)
        self.classificador.set("XGBoost")
        options = ["XGBoost", "MobileNet"]
        
        option_menu = tk.OptionMenu(self.frame_entrada, self.classificador, *options,command= lambda:self.mudar_classificador())
        option_menu.pack(side=tk.LEFT, padx=(100,5),pady=4)


            # Botão para treinar o modelo
        self.modelo = None
        train_button = tk.Button(self.frame_entrada, text="Treinar e usar Modelo",command=lambda:self.treinar_modelo())
        train_button.pack(side=tk.LEFT, padx=5)
            # Botão para usar o modelo ja treinado
        get_modelo_button = tk.Button(self.frame_entrada, text="Usar Modelo ja treinado",command=lambda:self.carregar_modelo())
        get_modelo_button.pack(side=tk.LEFT, padx=5)

        if self.classificador.get() == 'XGBoost':
            print("a")

        elif self.classificador.get()  == 'MobileNet':
            print("a")

    def mudar_classificador(self):
        if self.classificador.get() == "XGBoost":
            self.classificador.set("MobileNet")
        else:
            self.classificador.set("XGBoost")

    def treinar_modelo(self):
        
        if self.classificador.get()  == 'MobileNet':
            carregar_modelo_nao_treinado()
        else:
            classificar_xgboost()
            


    def carregar_modelo(self):
        
        if self.classificador.get()  == 'MobileNet':
            carregar_modelo_treinado()
        else:
            classificar_xgboost()


# --------------------------------------------------------------------------------
# Arquivo: C:\Users\Leonardo Loureiro\OneDrive\Documentos\Puc\PAI\G10\PAI_Trabalho_2024\Screens\Home.py
# --------------------------------------------------------------------------------

import tkinter as tk


class HomeScreen(BaseScreen):
    def __init__(self,manager, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #-----------------------------
        #       Configurações
        #-----------------------------

        self.title("Home")
        self.geometry("600x600")
        self.configure(bg='gray')

        #---------------------------
        #        Componentes
        #---------------------------

        # Label Titulo
        label = tk.Label(self, text="Home",font=("Helvetica", 20),pady=20,background="gray")
        label.pack()
        
        # Botão para mudar para a Screen Image
        buttonScreenImage = tk.Button(self, text="Visualizar Imagem",command=lambda:manager.change_to_ImageScreen())
        buttonScreenImage.pack(pady=10)

        # Botão para mudar para a Screen Roi
        buttonScreenRoi = tk.Button(self, text="Visualizar Roi",command=lambda:manager.change_to_RoiScreen())
        buttonScreenRoi.pack(pady=10)

        # Botão para mudar para a Screen Classificação
        buttonScreenRoi = tk.Button(self, text="Classificar Roi",command=lambda:manager.change_to_ClassificacaoScreen())
        buttonScreenRoi.pack(pady=10)
    


# --------------------------------------------------------------------------------
# Arquivo: C:\Users\Leonardo Loureiro\OneDrive\Documentos\Puc\PAI\G10\PAI_Trabalho_2024\Screens\Image.py
# --------------------------------------------------------------------------------

import tkinter as tk
from tkinter import filedialog, messagebox

import scipy
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import os

class ImageScreen(BaseScreen):
    def __init__(self,manager ,*args, **kwargs):
        super().__init__(*args, **kwargs)

    
        self.numImagem = tk.IntVar(self,0)
        self.numPessoa = tk.IntVar(self,0)

        #-----------------------------
        #       Configurações
        #-----------------------------

        self.title("Image Screen")
        self.geometry("800x600")
        self.configure(bg='gray')

        #---------------------------
        #        Componentes
        #---------------------------
 
        # Menu para selecionar imagens

        self.frame_entrada = tk.Frame(self)
        self.frame_entrada.pack(side=tk.TOP, pady=10, fill=tk.X)
            
            # Labels para a entrada do número da imagem
        self.numPessoaLabel = tk.Label(self.frame_entrada,text="Pessoa: ")
        self.numPessoaLabel.pack(side=tk.LEFT, padx=5)

        self.numPessoaEntry = tk.Entry(self.frame_entrada, textvariable= str(self.numPessoa))
        self.numPessoaEntry.pack(side=tk.LEFT, padx=5)
        self.numPessoa.trace_add("write", self.alterar_numPessoa)
            
        self.numImagemLabel = tk.Label(self.frame_entrada,text="Imagem: ")
        self.numImagemLabel.pack(side=tk.LEFT, padx=5)

        self.numImagemEntry = tk.Entry(self.frame_entrada, textvariable= str(self.numImagem))
        self.numImagemEntry.pack(side=tk.LEFT, padx=5)

        self.numImagem.trace_add("write", self.alterar_numImagem)

            # Botões para selecionar a imagem, passar e voltar
        selecionar_imagem_button = tk.Button(self.frame_entrada, text="Selecionar Imagem", command=lambda: self.selecionar_imagem())
        selecionar_imagem_button.pack(side=tk.LEFT, padx=5,pady=4)

        voltar_matimagem_button = tk.Button(self.frame_entrada, text="<",command= lambda: self.numImagem.set(self.numImagem.get()-1))
        voltar_matimagem_button.pack(side=tk.LEFT, padx=5)

        passar_matimagem_button= tk.Button(self.frame_entrada, text=">",command= lambda: self.numImagem.set(self.numImagem.get()+1))
        passar_matimagem_button.pack(side=tk.LEFT, padx=5)

            #Zoom
        self.zoomLabel = tk.Label(self.frame_entrada,text="Zoom: ")
        self.zoomLabel.pack(side=tk.LEFT,padx=(5,1))

        zoomAdd_button = tk.Button(self.frame_entrada, text="+",command= lambda: self.zoom_image(zoom=1.25) )
        zoomAdd_button.pack(side=tk.LEFT, padx=(0,5))

        zoomRemove_button = tk.Button(self.frame_entrada, text="-",command= lambda: self.zoom_image(zoom=0.8) )
        zoomRemove_button.pack(side=tk.LEFT, padx=5)


        # Canvas da Imagem
        self.frame_imagem = tk.Frame(self)
        self.frame_imagem.pack()

        self.canvas = tk.Canvas(self.frame_imagem,width=636,height=434)
        self.canvas.grid(column=0,row=0)
        self.canvas.bind("<Button-1>", self.selecionar_roi)

        self.retangulo_figado = None
        self.retangulo_rim = None
        self.is_rim = False

            # Barras de rolagem
        self.scroll_y = tk.Scrollbar(self.frame_imagem, orient="vertical", command=self.canvas.yview)
        self.scroll_y.grid(column=1,row=0,sticky="nsew")
        
        self.scroll_x = tk.Scrollbar(self.frame_imagem, orient="horizontal", command=self.canvas.xview)
        self.scroll_x.grid(column=0,row=1,sticky="nsew")

        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)

        # Label do indice Hepatorenal
        self.indiceHepatorenal = tk.StringVar(self,"Índice Hepatorenal HI: ")
        self.indice_hepatorenal_Label = tk.Label(self, textvariable= self.indiceHepatorenal,background="gray")
        self.indice_hepatorenal_Label.pack(padx=(0,5))

        # Menu com botões
        self.button_frame = tk.Frame(self,background="gray")
        self.button_frame.pack(side=tk.BOTTOM, pady=10)

        histogramas_imagem_button = tk.Button(self.button_frame, text="Histogramas da Imagem",command=lambda: self.show_histogram() )
        histogramas_imagem_button.pack(side=tk.LEFT, padx=(0,5))

        salvar_rois_button = tk.Button(self.button_frame, text="Salvar ROI's",command=lambda: self.salvar_rois())
        salvar_rois_button.pack(side=tk.LEFT, padx=(0,5))

        abrir_rois_button = tk.Button(self.button_frame, text="Abrir ROI's",command=lambda:manager.change_to_RoiScreen(imagem_rim = self.roi_rim,imagem_figado = self.roi_figado) )
        abrir_rois_button.pack(side=tk.LEFT, padx=(0,5))



    
        

    def alterar_mat_imagem(self):
            imagens_data = self.pessoas_data[self.numPessoa.get()] #Pessoa
            imagens_data = imagens_data[self.numImagem.get()] #Exame

            self.imagem = Image.fromarray(imagens_data)
            self.tkimage = ImageTk.PhotoImage(self.imagem)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW,image = self.tkimage)
            self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
    def alterar_numPessoa(self,*args):
        if self.numPessoa.get() >= 55:
            self.numPessoa.set(0)
            self.numImagem.set(0)
        elif self.numPessoa.get() < 0:
            self.numPessoa.set(54)
            self.numImagem.set(9)

        self.alterar_mat_imagem()

    def alterar_numImagem(self,*args):
        if self.numImagem.get() >= 10:
            self.numImagem.set(0)
            self.numPessoa.set(self.numPessoa.get()+1)
        elif self.numImagem.get() < 0:
            self.numImagem.set(9)
            self.numPessoa.set(self.numPessoa.get()-1)
        
        self.alterar_mat_imagem()

    def selecionar_imagem(self):
        tipo_imagem = messagebox.askquestion("Tipo de arquivo", "A imagem está em um arquivo .mat?")
        if tipo_imagem == 'yes':
            mat_file_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
            mat_data = scipy.io.loadmat(mat_file_path)
            data = mat_data['data']
            self.pessoas_data = data['images'][0]

            self.alterar_mat_imagem()
            
           
        else:
            image_file_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
            self.imagem = Image.open(image_file_path)
            self.tkimage = ImageTk.PhotoImage(self.imagem)
            self.canvas.create_image(0, 0, anchor=tk.NW,image = self.tkimage)
            
    def zoom_image(self,zoom):
        width, height = self.imagem.size
        new_size = (int(width * zoom), int(height * zoom))
        self.imagem = self.imagem.resize(new_size)
        self.tkimage = ImageTk.PhotoImage(self.imagem)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW,image = self.tkimage)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def selecionar_roi(self, event):

        # Salva as coordenadas do mouse
        x_mouse = event.x
        y_mouse = event.y

        print(x_mouse)
        print(y_mouse)
        # Gera os retangulos de seleção das ROI's na imagem
        texto = ""
        if self.is_rim:
            texto = "Rim"
            if self.retangulo_rim:
                self.canvas.delete(self.retangulo_rim)
                self.canvas.delete(self.texto_rim)

            self.x_sup_rim = (x_mouse - 12)
            self.y_sup_rim = (y_mouse - 12)

            self.retangulo_rim = self.canvas.create_rectangle((x_mouse - 12), (y_mouse - 12), (x_mouse + 12), (y_mouse + 12), outline="green",width=2)
            self.texto_rim = self.canvas.create_text((x_mouse - 12), (y_mouse - 22),text=texto, fill="green", font=("Arial", 8, "bold"))
            self.is_rim = False
            self.roi_rim = self.imagem.crop(((x_mouse - 12), (y_mouse - 12), (x_mouse + 12), (y_mouse + 12)))

            self.indiceHepatorenal.set("Índice Hepatorenal HI: "+str(self.calc_indice_hepatorenal()))
        else:
            texto = "Fígado"
            if self.retangulo_figado:
                self.canvas.delete(self.retangulo_figado)
                self.canvas.delete(self.texto_figado)
                self.canvas.delete(self.retangulo_rim)
                self.canvas.delete(self.texto_rim)

            self.x_sup_figado = (x_mouse - 12)
            self.y_sup_figado = (y_mouse - 12)
            self.retangulo_figado = self.canvas.create_rectangle((x_mouse - 12), (y_mouse - 12), (x_mouse + 12), (y_mouse + 12), outline="green",width=2)
            self.texto_figado = self.canvas.create_text((x_mouse - 12), (y_mouse - 22),text=texto, fill="green", font=("Arial", 8, "bold"))
            self.is_rim = True
            self.roi_figado = self.imagem.crop(((x_mouse - 12), (y_mouse - 12), (x_mouse + 12), (y_mouse + 12)))
            
            
        
        # Funções de teste
        #cropped_image = self.imagem.crop(((x_mouse - 12), (y_mouse - 12), (x_mouse + 12), (y_mouse + 12)))
        #cropped_image.show()  # Exibe a imagem recortada em uma nova janela

    def salvar_rois(self):
        
        # Ajusta os tons de cinza da ROI do Fígado
        array = np.array(self.roi_figado)
        array = np.clip(array * self.calc_indice_hepatorenal(), 0, 255).astype(np.uint8)
        nova_roi = Image.fromarray(array)

        # Salva a ROI do figado no diretorio
        diretorio = os.path.join("PAI_Trabalho_2024", "Database", "ROIS_figado")
        nome_arquivo = "ROI_"+str(self.numPessoa.get())+"_"+str(self.numImagem.get())+".png" 
        caminho = os.path.join(diretorio,nome_arquivo)
        nova_roi.save(caminho)

        # Salva a ROI no csv
        create_coordenadas(nome_arquivo=nome_arquivo,
               xSUP_figado=self.x_sup_figado,
               ySUP_figado=self.y_sup_figado,
               xSUP_rim=self.x_sup_rim,
               ySUP_rim=self.y_sup_rim,
               HI = self.calc_indice_hepatorenal(),
               paciente=self.numPessoa.get())
            
        #GLCM

        glcms_results,glcms = calc_glcm(nova_roi)

        #HARALICK
        haralick_entropia = {}
        haralick_homogeneidade = {}

        for i,glcm in glcms.items():
            haralick_entropia[i] = calc_haralick_entropia(glcm)
            haralick_homogeneidade[i] = calc_haralick_homogeneidade(glcm)


        

        #Tamura
        tamura = calc_tamura(nova_roi)

        create_caracteristicas(nome_arquivo=nome_arquivo,paciente=self.numPessoa.get()
                               ,glcms_results=glcms_results,
                               haralick_entropia=haralick_entropia,
                               haralick_homogeneidade=haralick_homogeneidade,tamura=tamura)

   
        

    def calc_indice_hepatorenal(self):
    
        array = np.array(self.roi_figado)
        media_figado = np.mean(array)

        array = np.array(self.roi_rim)
        media_rim = np.mean(array)

        return (media_figado / media_rim)
    
    def show_histogram(self):    
        array = np.array(self.imagem)
        histograma, limites = np.histogram(array, bins=256, range=(0, 255))
            
        plt.figure()
        plt.ylabel("Quantidade")
        plt.xlabel("Valor do Pixel")
            
        plt.plot(limites[0:-1], histograma)
        plt.xlim(0, 255)
        plt.show()

# --------------------------------------------------------------------------------
# Arquivo: C:\Users\Leonardo Loureiro\OneDrive\Documentos\Puc\PAI\G10\PAI_Trabalho_2024\Screens\Manager.py
# --------------------------------------------------------------------------------




class ScreenManager:
    def __init__(self):
        self.tela = HomeScreen(manager=self)
        self.tela.mainloop()

    def change_to_ImageScreen(self):
        self.tela.destroy()
        self.tela = ImageScreen(manager=self)

    def change_to_HomeScreen(self):
        self.tela.destroy()
        self.tela = HomeScreen(manager=self)

    def change_to_RoiScreen(self,imagem_rim = None,imagem_figado = None):
        self.tela.destroy()
        self.tela = RoiScreen(imagem_rim=imagem_rim,imagem_figado=imagem_figado)
    
    def change_to_ClassificacaoScreen(self):
        self.tela.destroy()
        self.tela = ClassificacaoScreen(manager=self)



# --------------------------------------------------------------------------------
# Arquivo: C:\Users\Leonardo Loureiro\OneDrive\Documentos\Puc\PAI\G10\PAI_Trabalho_2024\Screens\Roi.py
# --------------------------------------------------------------------------------

import tkinter as tk
from tkinter import filedialog, messagebox

import scipy
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from pathlib import Path
import sys



class RoiScreen(BaseScreen):
    def __init__(self,imagem_rim = None,imagem_figado = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if imagem_rim and imagem_figado:
            self.imagem_rim = imagem_rim.resize((240,240))
            self.imagem_figado = imagem_figado.resize((240,240))

        #-----------------------------
        #       Configurações
        #-----------------------------

        self.title("Roi Screen")
        self.geometry("800x400")
        self.configure(bg='gray')

        #---------------------------
        #        Componentes
        #---------------------------
 
        # Menu para selecionar imagens

        self.frame_entrada = tk.Frame(self)
        self.frame_entrada.pack(side=tk.TOP, pady=10, fill=tk.X)
            
            # Labels para a entrada do número da imagem
        self.numPessoaLabel = tk.Label(self.frame_entrada,text="Pessoa: ")
        self.numPessoaLabel.pack(side=tk.LEFT, padx=5)

        self.numImagemLabel = tk.Label(self.frame_entrada,text="Imagem: ")
        self.numImagemLabel.pack(side=tk.LEFT, padx=5)

            # Botões para selecionar a imagem, passar e voltar
        selecionar_imagem_button = tk.Button(self.frame_entrada, text="Selecionar Imagem", command=lambda: self.selecionar_imagem())
        selecionar_imagem_button.pack(side=tk.LEFT, padx=5,pady=4)

        # Option Menu para ROI de figado ou rim
        selected_option = tk.StringVar(self)
        selected_option.set("Rim")
        options = ["Rim", "Figado"]
        self.is_rim = True
        option_menu = tk.OptionMenu(self.frame_entrada, selected_option, *options, command=self.mudar_roi)
        option_menu.pack(side=tk.LEFT, padx=5,pady=4)

            #Zoom
        self.zoomLabel = tk.Label(self.frame_entrada,text="Zoom: ")
        self.zoomLabel.pack(side=tk.LEFT,padx=(5,1))

        zoomAdd_button = tk.Button(self.frame_entrada, text="+",command= lambda: self.zoom_image(zoom=1.25) )
        zoomAdd_button.pack(side=tk.LEFT, padx=(0,5))

        zoomRemove_button = tk.Button(self.frame_entrada, text="-",command= lambda: self.zoom_image(zoom=0.8) )
        zoomRemove_button.pack(side=tk.LEFT, padx=5)


        # Canvas da Imagem
        self.frame_imagem = tk.Frame(self)
        self.frame_imagem.pack()


        self.canvas = tk.Canvas(self.frame_imagem,width=240,height=240)
        self.canvas.grid(column=0,row=0)
        if self.imagem_rim:
            self.imagem = self.imagem_rim
            self.tkimage = ImageTk.PhotoImage(self.imagem)
            self.canvas.create_image(0, 0, anchor=tk.NW,image = self.tkimage)
        
            # Barras de rolagem
        self.scroll_y = tk.Scrollbar(self.frame_imagem, orient="vertical", command=self.canvas.yview)
        self.scroll_y.grid(column=1,row=0,sticky="nsew")
        
        self.scroll_x = tk.Scrollbar(self.frame_imagem, orient="horizontal", command=self.canvas.xview)
        self.scroll_x.grid(column=0,row=1,sticky="nsew")

        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)

        # Label do indice Hepatorenal
        self.indiceHepatorenal = tk.StringVar(self,"Índice Hepatorenal HI: ")
        self.indice_hepatorenal_Label = tk.Label(self,background="gray")
        self.indice_hepatorenal_Label.pack(padx=(0,5))

        # Menu com botões
        self.button_frame = tk.Frame(self,background="gray")
        self.button_frame.pack(side=tk.BOTTOM, pady=10,padx=(10,10))

        histogramas_imagem_button = tk.Button(self.button_frame, text="Histogramas da Imagem",command=lambda: self.show_histogram() )
        histogramas_imagem_button.pack(side=tk.LEFT, padx=(0,5))

        calcular_caracteristicas_button = tk.Button(self.button_frame, text="Calcular Características", command=lambda: self.mostrar_caracteristicas())
        calcular_caracteristicas_button.pack(side=tk.LEFT, padx=5)
        
    def selecionar_imagem(self):
        image_file_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
        self.imagem = Image.open(image_file_path)
        self.tkimage = ImageTk.PhotoImage(self.imagem)
        self.canvas.create_image(0, 0, anchor=tk.NW,image = self.tkimage)
            
    def zoom_image(self,zoom):
        width, height = self.imagem.size
        new_size = (int(width * zoom), int(height * zoom))
        self.imagem = self.imagem.resize(new_size)
        self.tkimage = ImageTk.PhotoImage(self.imagem)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW,image = self.tkimage)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
    
    def show_histogram(self):    
        array = np.array(self.imagem)
        histograma, limites = np.histogram(array, bins=256, range=(0, 255))
            
        plt.figure()
        plt.ylabel("Quantidade")
        plt.xlabel("Valor do Pixel")
            
        plt.plot(limites[0:-1], histograma)
        plt.xlim(0, 255)
        plt.show()

    def mudar_roi(self, *args):
        if self.is_rim:
            self.is_rim = False
            self.imagem = self.imagem_figado

            self.tkimage = ImageTk.PhotoImage(self.imagem)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW,image = self.tkimage)

        elif self.imagem is self.imagem_figado:
            self.imagem = self.imagem_rim
            self.is_rim = True

            self.tkimage = ImageTk.PhotoImage(self.imagem)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW,image = self.tkimage)
       
    def mostrar_caracteristicas(self):
        modal = tk.Toplevel(self)
        modal.title("Características Calculadas")
        modal.geometry("700x700")
        
        titulo_caracteristica = tk.Label(modal, text="Caracteristicas", font=("Helvetica", 12, "bold"))
        titulo_caracteristica.pack(pady=10)

        imagem = self.imagem_rim
        if not self.is_rim:
            imagem = self.imagem_figado
            

        #GLCM

        glcms_results,glcms = calc_glcm(imagem)

        text_glcm = "Distancias d = 1, 2, 4, 8:\n"
        for d, props in glcms_results.items():
            text_glcm += f"\nDistancia {d}:\n"
            for prop, valor in props.items():
                text_glcm += f"{prop}: {valor:.4f}\n"


        #HARALICK
        text_haralick = "HARALICK:\n"
        for i,glcm in glcms.items():
            
            text_haralick += f"(GLCM {i})\n"
            text_haralick += "Entropia: " + str(calc_haralick_entropia(glcm)) + "\n"
            text_haralick += "Homogeneidade: " + str(calc_haralick_homogeneidade(glcm)) + "\n"

        

        #Tamura
        coarseness,contrast,directionality,line_likeness,regularity,roughness = calc_tamura(imagem)
        text_tamura = "TAMURA: \n"
        text_tamura += f"Coarseness: {coarseness}\n"
        text_tamura += f"Contrast: {contrast}\n"
        text_tamura += f"Directionality: {directionality}\n"
        text_tamura += f"Line Likeness: {line_likeness}\n"
        text_tamura += f"Regularity: {regularity}\n"
        text_tamura += f"Roughness: {roughness}\n" 
   
        

        #Plotagem
        GLCM_Label = tk.Label(modal, text=text_glcm, borderwidth=2, relief="solid")
        GLCM_Label.pack(side=tk.LEFT, padx=10, pady=10)

        Haralick_Label = tk.Label(modal, text=text_haralick, borderwidth=2, relief="solid")
        Haralick_Label.pack(side=tk.LEFT, padx=10, pady=10)

        Tamura_Label = tk.Label(modal, text=text_tamura, borderwidth=2, relief="solid")
        Tamura_Label.pack(side=tk.LEFT, padx=10, pady=10)


# --------------------------------------------------------------------------------
# Arquivo: C:\Users\Leonardo Loureiro\OneDrive\Documentos\Puc\PAI\G10\PAI_Trabalho_2024\main.py
# --------------------------------------------------------------------------------
print(tf. __version__)
a = ScreenManager()

