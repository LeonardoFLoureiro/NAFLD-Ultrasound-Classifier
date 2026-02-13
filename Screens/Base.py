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