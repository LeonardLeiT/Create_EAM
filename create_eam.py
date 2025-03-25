#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python version of the code Zhou04_create_v2.f
original author: X. W. Zhou, xzhou@sandia.gov
based on updates by: Lucas Hale lucas.hale@nist.gov
written by: Germain Clavier g.m.g.c.clavier@tue.nl
This script requires the numpy library.
"""

import sys
import argparse as ap
from datetime import date

import numpy as np
from eamDatabase import Database
from eamDatabase import PairData

# Function to retrieve pair data
def get_pair_data(element1, element2):
    key = tuple(sorted([element1, element2]))  # Sort to ensure order independence
    return PairData.get(key, None)  # Return None if pair not found

def H(x):
    return np.where(x >= 0, 1, 0)

# fi(rij)
def prof(at, r):
    atom = Database[at]
    f = np.zeros(r.shape)
    # Apply condition to entire function
    mask = r >= 0.5 
    numerator = atom.fe * np.exp(-atom.beta * (r[mask] / atom.re - 1.0))
    denominator = 1.0 + (r[mask] / atom.re - atom.lambd) ** atom.n  
    f[mask] = numerator / denominator
    return f

# phi(r)
def pair(at1, at2, r):  
    if at1 == at2:
        atom = Database[at1]
    else:
        atom = get_pair_data(at1, at2)
    psi1 = atom.A * np.exp(-atom.alpha * (r / atom.re - 1.0))
    psi1 /= (1.0 + ((r / atom.re - atom.kappa) ** atom.m) * H(r / atom.re - atom.kappa))
    psi2 = atom.B * np.exp(-atom.beta * (r / atom.re - 1.0))
    psi2 /= (1.0 + ((r / atom.re - atom.lambd) ** atom.n) * H(r / atom.re - atom.lambd))
    psi = psi1 - psi2
    return psi

# F(P)
def embed(at, rho):  
    atom = Database[at]
    emb = np.zeros(rho.shape)
    rhon = 0.85 * atom.rhoe
    rhoo = 1.15 * atom.rhoe
    for i, r in enumerate(rho):
        if r == 0:
            emb[i] = 0
        elif r < rhon:
            dr = r / rhon - 1
            emb[i] = atom.Fn0 + atom.Fn1 * dr + atom.Fn2 * dr**2 + atom.Fn3 * dr**3
        elif r < atom.rhoe:
            dr = r / atom.rhoe - 1
            emb[i] = atom.F0 + atom.F1 * dr + atom.F2 * dr**2 + atom.F3_1 * dr**3
        elif r < rhoo:
            dr = r / atom.rhoe - 1
            emb[i] = atom.F0 + atom.F1 * dr + atom.F2 * dr**2 + atom.F3_2 * dr**3
        else:
            dr = r / atom.rhos
            emb[i] = atom.Fe * (1.0 - atom.eta * np.log(dr)) * dr**atom.eta
    return emb

def write_file(attypes, filename, Fr, rhor, z2r, nrho, drho, nr, dr, rc):
    struc = "bcc"
    with open(filename, "w") as f:
        f.write("DATE: " + date.today().strftime("%Y-%m-%d") + " UNITS: metal")
        f.write(" CONTRIBUTOR: Ting Lei, leiting@imech.ac.cn\n")
        f.write(" Reference: Wu Y, Yu W, Shen S, Materials & Design, 2023, 230: 111999.\n")
        f.write(" Reference: X. W. Zhou, R. A. Johnson, H. N. G. Wadley, Phys. Rev. B, 69, 144113(2004)\n")
        f.write("{:<5d} {:<24}\n".format(len(attypes), " ".join(attypes)))
        f.write("{:<5d} {:<24.16e} {:<5d} {:<24.16e} {:<24.16e}\n".format(nrho, drho, nr, dr, rc))
        for at in attypes:
            atom = Database[at]
            f.write(
                "{:>5d} {:>15.5f} {:>15.5f} {:>8}\n".format(
                    atom.ielement, atom.amass, atom.blat, struc
                )
            )
            for i, fr in enumerate(Fr[at]):
                f.write(" {:>24.16E}".format(fr))
                if not (i + 1) % 5:
                    f.write("\n")
            for i, rho in enumerate(rhor[at]):
                f.write(" {:>24.16E}".format(rho))
                if not (i + 1) % 5:
                    f.write("\n")
        for n1 in range(len(attypes)):
            for n2 in range(n1 + 1):
                for i, z in enumerate(z2r[n1, n2]):
                    f.write(" {:>24.16E}".format(z))
                    if not (i + 1) % 5:
                        f.write("\n")

def create_eam(atnames, nr=2000, nrho=2000):
    if not atnames:
        raise ValueError("Element names must be provided.")

    for n in atnames:
        if n not in Database:
            valid_elements = " ".join(Database.keys())
            raise ValueError(f"Element {n} not found in database. Supported elements are: {valid_elements}")

    ntypes = len(atnames)
    outfilename = "".join([*atnames, ".eam.alloy"])
    rhor = {}
    Fr = {}

    alatmax = max([Database[at].re for at in atnames])
    rhoemax = max([Database[at].rhoe for at in atnames])
    rc = np.sqrt(2) * alatmax ## BCC Third Nearest Neighbor
    rst = 0.5
    r = np.linspace(0.0, rc, num=nr, dtype=np.double)
    dr = r[1] - r[0]
    r[r < rst] = rst
    z2r = np.zeros([ntypes, ntypes, nr])
    rhomax = -np.inf

    for i, n1 in enumerate(atnames):
        for j, n2 in enumerate(atnames):
            if j > i:
                continue
            if i == j:
                rhor[n1] = prof(n1, r)
                rhomax = max(rhomax,np.max(rhor[n1]))
                z2r[i, j, :] = r * pair(n1, n2, r)
            else:
                z2r[i, j, :] = r * pair(n1, n2, r)
    z2r = np.where(z2r, z2r, z2r.transpose((1, 0, 2)))
    rhomax = max(rhomax, 2.0 * rhoemax, 100.0)
    rho = np.linspace(0.0, rhomax, num=nrho, dtype=np.double)
    drho = rho[1] - rho[0]
    for i, n1 in enumerate(atnames):
        Fr[n1] = embed(n1, rho)

    write_file(atnames, outfilename, Fr, rhor, z2r, nrho, drho, nr, dr, rc)

if __name__ == "__main__":
    try:
        elements = ["Hf", "Nb", "Ta", "Ti", "Zr"]  # Properly define elements as strings  
        create_eam(elements)  
    except KeyboardInterrupt as exc:
        raise SystemExit("User interruption.") from exc
