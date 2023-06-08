# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:33:44 2022

@author: RZ
"""

import numpy as np
import warnings
from numba import njit

#suppress warnings
warnings.filterwarnings('ignore')

@njit(parallel=True, fastmath=True)
def randX(a, b):
    P = np.random.rand(a, b)
    return P

@njit(parallel=True, fastmath=True)
def randn(a, b):
    w = randX(a, b) * 2.0 - 1.0
    return w

@njit(parallel=True, fastmath=True)
def normRows(a):
    P = a.copy()
    rows, columns = a.shape
    for x in range(0, rows):
        sumSq = 0
        for y in range(0, columns):
            v = P[x, y]
            sumSq += v ** 2.0
            len = np.sqrt(sumSq)
            for y in range(0, columns):
                P[x, y] = P[x, y] / len
    return P

@njit(parallel=True, fastmath=True)
def sumsqr(a):
    rows, columns = a.shape
    sumSq = 0
    for x in range(0, rows):
        for y in range(0, columns):
            v = a[x, y]
            sumSq += v ** 2.0
    return sumSq

@njit(parallel=True, fastmath=True)
def rands(a, b):
    #  Mnożenie tych liczb przez 2.0 i odejmowanie 1.0 skutkuje losowymi liczbami z zakresu od -1.0 do 1.0.
    w = randX(a, b) * 2.0 - 1.0
    b = randX(a, 1) * 2.0 - 1.0
    return w, b

@njit(parallel=True, fastmath=True)
def nwtan(s, p):
    # Optymalne rozlozenie wag na start
    magw = 0.7 * s ** (1.0 / p)
    # Stworzenie macierza wag wejscie na wyjscie - s wiersze p kolumny
    w = magw * normRows(randn(s,p))\
    # Tworzy macierz biasów z randn czyli randomowymi numerami bazujac na rozkladzie normalnym
    b = magw * randn(s,1)
    rng = np.zeros((1, p))
    rng = rng + 2.0
    mid = np.zeros((p, 1))
    # Sprawia, że wartości macierzy w są bliskie 1
    w = 2.0 * w / np.dot(np.ones((s,1)), rng)
    # W przypadku mid wypelnionego zerami nic sie nie zmienia ale gdyby byly tam losowe wartosci
    # to wprowadzaloby to dodatkowe zmiany wartosci
    b = b - np.dot(w, mid)
    return w, b

@njit(parallel=True, fastmath=True)
def nwlog(s, p):
    magw = 2.8 * s ** (1.0 / p)
    w = magw * normRows(randn(s,p))
    b = magw * randn(s,1)
    rng = np.zeros((1, p))
    rng = rng + 2.0
    mid = np.zeros((p, 1))
    w = 2.0 * w / np.dot(np.ones((s,1)), rng)
    b = b - np.dot(w, mid)
    return w, b

@njit(parallel=True, fastmath=True)
def tansig(n, b):
    n = n + b
    a = 2.0 / (1.0 + np.exp(-2.0 * n)) - 1.0
    rows, columns = a.shape
    for x in range(0, rows):
        for y in range(0, columns):
            v = a[x, y]
            # Rozwiazuje problem zwiazany z nieskonczonymi wartosciami wynikajcymi z funkcji ekspotencjalnej
            if np.abs(v) == np.inf:
                a[x, y] = np.sign(n[x, y])
    return a

@njit(parallel=True, fastmath=True)
def logsig(n, b):
    n = n + b
    a = 1.0 / (1.0 + np.exp(-n))
    rows, columns = a.shape
    for x in range(0, rows):
        for y in range(0, columns):
            v = a[x, y]
            if np.abs(v) == np.inf:
                a[x, y] = np.sign(n[x, y])
    return a

@njit(parallel=True, fastmath=True)
def purelin(n, b):
    a = n + b
    return a

@njit(parallel=True, fastmath=True)
def deltatan(a, d, *w):
    if not w:
        d = (1.0 - (a * a)) * d
    else:
        d = (1.0 - (a * a)) * np.dot(np.transpose(w[0]), d)
    return d

@njit(parallel=True, fastmath=True)
def deltalog(a, d, *w):
    if not w:
        d = a * (1.0 - a) * d
    else:
        d = a * (1.0 - a) * np.dot(np.transpose(w[0]), d)
    return d

@njit(parallel=True, fastmath=True)
def deltalin(a, d):
    return d

@njit(parallel=True, fastmath=True)
def learnbp(p, d, lr):
    x = lr * d
    dw = np.dot(x, np.transpose(p))
    Q = p.shape[1]
    db = np.dot(x, np.ones((Q, 1)))
    return dw, db
