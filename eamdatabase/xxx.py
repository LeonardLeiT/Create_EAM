import sys
import argparse as ap
from datetime import date
import numpy as np
from eamDatabase import Database

def prof(at, r):
    atom = Database[at]
    f = np.zeros(r.shape)
    for i, ri in enumerate(r):
        if ri < atom.r2e:
            dr = atom.re / ri
            f[i] = atom.fe * (dr ** atom.beta)
        elif ri < atom.rc:
            dr = ri / atom.r2e - 1
            f[i] = atom.m3 * (dr ** 3) + atom.m2 * (dr ** 2) + atom.m1 * dr + atom.m0
        else:
            f[i] = 2.77555756e-17
    return f

def pair1(at, r):
    atom = Database[at]
    pis = np.zeros(r.shape)
    for i, ri in enumerate(r):
        if ri < atom.re:
            dr = ri / atom.re - 1
            pis_m = atom.k3 * (dr ** 3) + atom.k2 * (dr ** 2) + atom.k1 * dr + atom.k0
            pis[i] = pis_m + atom.ka * (pis_m - atom.k0) * (dr ** 2)
        elif ri < atom.r2e:
            dr = ri / atom.re - 1
            pis[i] = atom.k3 * (dr ** 3) + atom.k2 * (dr ** 2) + atom.k1 * dr + atom.k0
        elif ri < atom.rc:
            dr = ri / atom.r2e - 1
            pis[i] = atom.l3 * (dr ** 3) + atom.l2 * (dr ** 2) + atom.l1 * dr + atom.l0
        else:
            pis[i] = 1.66533454e-16
    return pis

def pair(at1, at2, r):
    if at1 == at2:
        psi = pair1(at1, r)
    else:
        psia = pair1(at1, r)
        psib = pair1(at2, r)
        profa = prof(at1, r)
        profb = prof(at2, r)
        psi = 0.5 * (profb / profa * psia + profa / profb * psib)
    return psi

def embed(at, rho):
    atom = Database[at]
    emb = np.zeros(rho.shape)
    for i, rh in enumerate(rho):
        if rh == 0:
            emb[i] = 0
        else:
            dr = (rh / atom.pe) ** atom.n
            emb[i] = -(atom.Ec - atom.Ef) * (1.0 - np.log(dr)) * dr
    return emb

def write_file(attypes, filename, Fr, rhor, z2r, nrho, drho, nr, dr, rc):
    struc = "bcc"
    with open(filename, "w") as f:
        f.write("DATE: " + date.today().strftime("%Y-%m-%d") + " UNITS: metal BCC")
        f.write(" CONTRIBUTOR: Ting Lei leiting@imech.ac.cn\n")
        f.write("Reference: Bangwei Z, Yifang O. Physical review B, 1993.\n")
        f.write("Code Reference: Zhou X W, Johnson R A, Wadley H N G. Physical Review B, 2004.\n")
        f.write("{:<5d} {:<24}\n".format(len(attypes), " ".join(attypes)))
        f.write("{:<5d} {:<24.16e} {:<5d} {:<24.16e} {:<24.16e}\n".format(nrho, drho, nr, dr, rc))
        for at in attypes:
            atom = Database[at]
            f.write(
                "{:>5d} {:>15.5f} {:>15.5f} {:>8}\n".format(
                    atom.ielement, atom.Amass, atom.lattice, struc
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

def create_eam(argv=None):
    parser = ap.ArgumentParser(description="Script to create EAM alloy potential files.")

    parser.add_argument("-n", "--names", dest="name", nargs="+",
                        help="Element names")
    parser.add_argument("-nr", dest="nr", type=int, default=2000,
                        help="Number of point in r space [default 2000]")
    parser.add_argument("-nrho", dest="nrho", type=int, default=2000,
                        help="Number of point in rho space [default 2000]")
    args = parser.parse_args(argv)
    if not args.name:
        parser.print_help()
        sys.exit("")

    atnames = args.name
    nr = args.nr
    nrho = args.nrho

    for n in atnames:
        try:
            Database[n]
        except KeyError:
            output = "Element {} not found in database.\n".format(n)
            valid = "Supported elements are: {}".format(" ".join(Database.keys()))
            sys.exit("".join([output, valid]))

    ntypes = len(atnames)
    outfilename = "".join([*atnames, ".eam.alloy"])
    rhor = {}
    Fr = {}

    r3e = max([Database[at].rc for at in atnames])
    rhoemax = max([Database[at].pe for at in atnames])
    # r3e = np.sqrt(11.0) / 2.0 * alatmax
    rst = 0.4
    r = np.linspace(0.0001, r3e, num=nr, dtype=np.double)
    dr = r[1] - r[0]
    r[r < rst] = rst
    z2r = np.zeros([ntypes, ntypes, nr])
    # rhomax = -np.inf

    for i, n1 in enumerate(atnames):
        for j, n2 in enumerate(atnames):
            if j > i:
                continue
            if i == j:
                rhor[n1] = prof(n1, r)
                # rhomax = max(rhomax, np.max(rhor[n1]))
                z2r[i, j, :] = r * pair(n1, n2, r)
            else:
                z2r[i, j, :] = r * pair(n1, n2, r)
    z2r = np.where(z2r, z2r, z2r.transpose((1, 0, 2)))
    # rhomax = max(rhomax, rhoemax)
    rho = np.linspace(0.0, rhoemax * 3.5, num=nrho, dtype=np.double)
    drho = rho[1] - rho[0]
    for i, n1 in enumerate(atnames):
        Fr[n1] = embed(n1, rho)

    write_file(atnames, outfilename, Fr, rhor, z2r, nrho, drho, nr, dr, r3e)

if __name__ == "__main__":
    # e = 'Hf'
    # at = Database[e]
    # print(at)
    # print('VB:', at.Omiga*at.Bulk)
    # print('VG:', at.Omiga * at.G)
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # rho = np.linspace(0, at.pe * 5, num=100, dtype=np.double)
    # F = embed(e, rho)
    # plt.plot(rho, F)
    # # plt.plot(X, S)
    # plt.show()
    # r = np.linspace(0.5, at.rc + 0.1, num=100, dtype=np.double)
    # Fr = prof(e, r)
    # plt.plot(r, Fr)
    # # plt.plot(X, S)
    # plt.show()
    # r = np.linspace(0, at.rc, num=100, dtype=np.double)
    # Fr = pair1(e, r)
    # # r = r / at.re -1
    # plt.plot(r, Fr)
    # # plt.plot(X, S)
    # plt.show()
    try:
        create_eam(sys.argv[1:])
    except KeyboardInterrupt as exc:
        raise SystemExit("User interruption.") from exc

