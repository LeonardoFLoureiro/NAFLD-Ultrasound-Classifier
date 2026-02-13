# --------------------------------------------------------------------------------
# Arquivo: C:\Users\Leonardo Loureiro\OneDrive\Documentos\Puc\PAI\G10\PAI_Trabalho_2024\Screens\Classificacao.py
# --------------------------------------------------------------------------------

#NC = 1
#ND = 2
import tkinter as tk

from Analise_de_Imagens.Classificacao.MobileNet import carregar_modelo_nao_treinado, carregar_modelo_treinado
from Analise_de_Imagens.Classificacao.XGBoost import classificar_xgboost
from Screens.Base import BaseScreen


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
