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