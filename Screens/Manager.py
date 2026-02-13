# --------------------------------------------------------------------------------
# Arquivo: C:\Users\Leonardo Loureiro\OneDrive\Documentos\Puc\PAI\G10\PAI_Trabalho_2024\Screens\Manager.py
# --------------------------------------------------------------------------------




from Screens.Classificacao import ClassificacaoScreen
from Screens.Home import HomeScreen
from Screens.Image import ImageScreen
from Screens.Roi import RoiScreen


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