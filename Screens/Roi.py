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

from Analise_de_Imagens.Descricao.GLCM import calc_glcm
from Analise_de_Imagens.Descricao.Haralick import calc_haralick_homogeneidade
from Analise_de_Imagens.Descricao.Tamura import calc_tamura
from Screens.Base import BaseScreen
from codigo_consolidado import calc_haralick_entropia



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