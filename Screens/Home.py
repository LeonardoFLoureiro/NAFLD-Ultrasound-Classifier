# --------------------------------------------------------------------------------
# Arquivo: C:\Users\Leonardo Loureiro\OneDrive\Documentos\Puc\PAI\G10\PAI_Trabalho_2024\Screens\Home.py
# --------------------------------------------------------------------------------

import tkinter as tk

from Screens.Base import BaseScreen


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