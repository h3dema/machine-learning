"""

    conjunto de funcoes que lidam com as distribuicoes para MAB

"""
import numpy as np
from random import random
# from random import randint
from math import sqrt


def select(m, n):
    """
        retorna a mascara para selecionar aleatoriamente n elementos da matriz de m elementos
    """
    n = min(m, max(0, n))
    mask = [True for i in range(n)] + [False for i in range(m - n)]
    np.random.shuffle(mask)
    return mask


def return_random(T):
    """
        retorna T valores alaatórios
    """
    return [random() for i in range(T)]


def return_normal(T, mean=0.5, variance=sqrt(12)):
    """
        retorna T valores obtidos de uma distribuicao normal.
        variancia e media são passados como parametros.
    """
    return [abs(np.random.normal(loc=mean, scale=variance)) for i in range(T)]
