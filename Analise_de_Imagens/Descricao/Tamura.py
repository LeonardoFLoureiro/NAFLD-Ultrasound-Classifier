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