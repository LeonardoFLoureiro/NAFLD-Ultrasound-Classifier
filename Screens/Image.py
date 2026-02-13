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

from Analise_de_Imagens.Descricao.GLCM import calc_glcm
from Analise_de_Imagens.Descricao.Haralick import calc_haralick_homogeneidade
from Analise_de_Imagens.Descricao.Tamura import calc_tamura
from Database.CRUD_csv import create_caracteristicas, create_coordenadas
from Screens.Base import BaseScreen
from codigo_consolidado import calc_haralick_entropia


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