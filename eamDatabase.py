#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python version of the code Leonaed_Lei_create_v3.f
original author: X. W. Zhou, xzhou@sandia.gov
based on updates by: Ting Lei, leiting@imech.ac.cn

This file contains atom attributes for the EAM model and alloy combination. The
original file is EAM_code. It is designed to be used with create_eam.py script.
To add new contribution, just add new AtType instances.

Espercially for BCC alloy, this code is written for Hf/Nb/Ta/Ti/Zr/O system.
Reference Letter: Developing a variable charge potential for Hf/Nb/Ta/Ti/Zr/O system via machine learning global optimization
"""

import math

class AtType:
    def __init__(
        self,
        name,
        re,        # re(A)
        fe,        # fe
        rhoe,      # pe
        rhos,      # ps
        alpha,     # alpha
        beta,      # beta
        A,         # A (eV)
        B,         # B (eV)
        kappa,     # kappa
        lambd,    # lambda
        m,         # m
        n,         # n
        Fn0,       # Fn0(eV)
        Fn1,       # Fn1(eV)
        Fn2,       # Fn2(eV)
        Fn3,       # Fn3(eV)
        F0,        # F0(eV)
        F1,        # F1(eV)  
        F2,        # F2(eV)
        F3_1,      # F3-(eV)
        F3_2,      # F3+(eV)
        eta,       # eta(eV)
        Fe,        # Fe(eV)
        ielement,
        amass,
    ):
        self.name = name
        self.re = re
        self.fe = fe
        self.rhoe = rhoe
        self.rhos = rhos
        self.alpha = alpha
        self.beta = beta
        self.A = A
        self.B = B
        self.kappa = kappa
        self.lambd = lambd
        self.m = m
        self.n = n
        self.Fn0 = Fn0
        self.Fn1 = Fn1
        self.Fn2 = Fn2
        self.Fn3 = Fn3
        self.F0 = F0
        self.F1 = F1
        self.F2 = F2
        self.F3_1 = F3_1
        self.F3_2 = F3_2
        self.amass = amass
        self.eta = eta
        self.Fe = Fe
        self.ielement = ielement
        self.amass = amass
        self.blat = math.sqrt(3.0) * self.re / 2

    def __repr__(self):
        output = """{}:
        re = {}; fe = {};
        rhoe = {}; rhos = {};
        alpha = {}; beta = {};
        A = {}; B = {};
        kappa = {}; lambd = {};
        m = {}; n = {};
        Fn0 = {}; Fn1 = {}; Fn2 = {}; Fn3 = {}
        F0 = {}; F1 = {}; F2 = {}; F3_1 = {}; F3_2 = {}
        eta = {}; Fe = {};
        ielement = {}; amass = {};
        blat = {}; """
        return output.format(
            self.name,
            self.re,
            self.fe,
            self.rhoe,
            self.rhos,
            self.alpha,
            self.beta,
            self.A,
            self.B,
            self.kappa,
            self.lambd,
            self.m,
            self.n,
            self.Fn0,
            self.Fn1,
            self.Fn2,
            self.Fn3,
            self.F0,
            self.F1,
            self.F2,
            self.F3_1,
            self.F3_2,
            self.eta,
            self.Fe,
            self.ielement,
            self.amass,
            self.blat,
        )


Database = {}
"""
data from Yihan Wu et al. Developing a variable charge potential for Hf/Nb/Ta/Ti/Zr/O system via machine learning global optimization
Materials & Design 230 (2023) 111999
"""
Database["Hf"] = AtType(
    "Hf",           
    3.036695,
    2.279201,       
    39.447272,
    18.953257,
    7.340695,
    2.986907,
    0.626761,
    0.645988,
    0.440318,
    1.172855,
    38,
    21,
    -4.369507,
    -0.516980,
    0.710178,
    -3.142348,
    -4.412630,
    0,
    1.695061,
    -1.476944,
    1.534853,
    0.356582,
    -4.646960,
    72,
    178.49,
)

Database["Nb"] = AtType(
    "Nb",           
    2.841095,
    2.889829,       
    31.417267,
    31.699834,
    7.595435,
    4.417068,
    0.579206,
    0.909354,
    0.138183,
    0.368801,
    18,
    21,
    -4.924136,
    -0.535647,
    1.649732,
    -2.738757,
    -4.970108,
    0,
    1.928528,
    -0.764655,
    -0.765052,
    0.909898,
    -4.967403,
    41,
    92.90637,
)

Database["Zr"] = AtType(
    "Zr",           
    3.151684,
    2.196376,       
    38.378598,
    14.274940,
    7.456545,
    3.244183,
    0.499557,
    0.623267,
    0.370194,
    1.100718,
    26,
    23,
    -4.137022,
    -0.294409,
    0.938741,
    -2.904141,
    -4.165324,
    0,
    1.464487,
    1.377158,
    0.260959,
    0.278061,
    -4.399373,
    40,
    91.224,
)

Database["Ti"] = AtType(
    "Ti",           
    2.833527,
    1.953310,       
    32.274144,
    10.692859,
    6.939830,
    3.146787,
    0.627344,
    0.672419,
    0.438699,
    1.163775,
    21,
    23,
    -3.093380,
    -0.321162,
    0.496175,
    -2.276043,
    -3.120445,
    0,
    1.089671,
    -0.754620,
    0.239451,
    0.262594,
    -3.315911,
    22,
    47.88,
)

Database["Ta"] = AtType(
    "Ta",           
    2.791418,
    3.091057,       
    31.431176,
    28.577385,
    8.259481,
    4.143922,
    0.689443,
    1.019847,
    0.176531,
    0.390894,
    17,
    21,
    -5.326794,
    -0.579329,
    1.230484,
    -3.516980,
    -5.378197,
    0,
    2.310024,
    0.169535,
    2.618849,
    0.805663,
    -5.427809,
    73,
    180.9479,
)



class PairProperties:
    def __init__(self, re, alpha, beta, A, B, kappa, lambd, m, n):
        self.re = re
        self.alpha = alpha
        self.beta = beta
        self.A = A
        self.B = B
        self.kappa = kappa
        self.lambd = lambd
        self.m = m
        self.n = n

PairData = {
    tuple(sorted(["Hf", "Nb"])): PairProperties(2.969511, 8.756233, 3.076312, 0.410802, 0.573587, 1.062141, 1.207085, 28, 26),
    tuple(sorted(["Hf", "Zr"])): PairProperties(3.411035, 8.219280, 3.379865, 0.255156, 0.455267, 0.298418, 0.934017, 20, 26),
    tuple(sorted(["Hf", "Ti"])): PairProperties(3.229927, 7.929612, 3.149661, 0.289205, 0.486683, 0.803686, 0.975948, 20, 20),
    tuple(sorted(["Hf", "Ta"])): PairProperties(3.226335, 9.210666, 3.547917, 0.235810, 0.503209, 0.826245, 1.0753, 18, 20),
    tuple(sorted(["Nb", "Zr"])): PairProperties(2.998161, 8.519996, 3.162518, 0.384227, 0.593337, 1.011727, 1.175926, 24, 28),
    tuple(sorted(["Nb", "Ti"])): PairProperties(2.874570, 8.221129, 3.631985, 0.479955, 0.653086, 0.884104, 1.054186, 20, 20),
    tuple(sorted(["Nb", "Ta"])): PairProperties(2.617376, 7.704793, 3.610211, 0.934650, 1.147555, 0.249344, 0.477396, 18, 20),
    tuple(sorted(["Zr", "Ti"])): PairProperties(5.975460, 14.393860, 6.133182, 0.000420, 0.031419, 0.76071, 0.085286, 20, 20),
    tuple(sorted(["Zr", "Ta"])): PairProperties(4.697030, 12.892805, 6.033640, 0.004683, 0.074825, -0.279701, 0.993989, 18, 22),
    tuple(sorted(["Ti", "Ta"])): PairProperties(5.910024, 16.068098, 9.446161, 0.000177, 0.006090, -0.428474, 1.009379, 20, 20),
    tuple(sorted(["Hf", "O"])): PairProperties(1.938218, 7.899236, 3.162444, 0.983139, 1.574250, 0.464820, 1.028556, 20, 20),
    tuple(sorted(["Nb", "O"])): PairProperties(1.489420, 6.677822, 3.811131, 1.198669, 0.945410, 0.496161, 1.121436, 20, 20),
    tuple(sorted(["Zr", "O"])): PairProperties(1.966073, 6.885521, 3.461640, 0.959256, 1.486355, 0.526812, 1.083523, 20, 20),
    tuple(sorted(["Ti", "O"])): PairProperties(1.651121, 7.071813, 2.730021, 1.065502, 1.474285, 0.359030, 0.654934, 20, 20),
    tuple(sorted(["Ta", "O"])): PairProperties(1.985520, 6.569020, 3.931190, 1.173747, 1.842358, 0.621463, 0.868149, 20, 20),
    tuple(sorted(["O", "O"])): PairProperties(2.411822, 6.664614, 3.205781, 1.787134, 1.397276, 0.438147, 0.704935, 20, 20),
}