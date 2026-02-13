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