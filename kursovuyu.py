#!/usr/bin/env python3
import numpy
import itertools
import copy
import sys
import os
from numpy.linalg import solve


def delath_kursovuyu(variant):
    lambdas = variant["lambdas"]
    mu      = variant["mu"]
    Ns      = variant["Ns"]
    N1s     = variant["N1s"]
    S       = variant["S"]

    P = {True:[], False:[]}
    lambdamus = {True:[], False:[]}

    def get_lams(step, neogr):
        N, N1 = Ns[step], Ns[step] if neogr else N1s[step]+1
        return lambdas[step] * numpy.arange(N, N-N1, -1.)

    def get_mus(step, neogr):
        N = Ns[step] if neogr else N1s[step]+1
        PP = numpy.array([1]) if neogr or step==0 else P[neogr][step-1]
        multij = lambda i,j: numpy.minimum(i+1, numpy.maximum(S-j,0))
        mult = numpy.fromfunction(multij, (N, len(PP)))
        return (mu[step] * mult.dot(PP)).flatten()

    def compute_probabilities(la, mu):
        diag = -(numpy.concatenate((la, [0])) + numpy.concatenate(([0], mu)))
        mat = numpy.diag(diag)
        mat[numpy.fromfunction(lambda i,j:j==i-1, mat.shape)] = la # lower diagonal
        mat[numpy.fromfunction(lambda i,j:j==i+1, mat.shape)] = mu # upper diagonal
        eigenvals, eigenvects = numpy.linalg.eig(mat)
        idx = eigenvals.argsort()[::-1][0] # the index of the lowest eigenvalue
        solnn = eigenvects[:,idx]
        sol = solnn / numpy.sum(solnn)
        return sol

    for neogr in [False, True]:
        #print("%sogranichenii:" % ("ne" if neogr else "",))
        for step in range(0,len(Ns)):
            #print("step: %d" % (step,))
            lams = get_lams(step, neogr)
            mus = get_mus(step, neogr)
            lambdamus[neogr].append({"lambda":lams, "mu":mus})
            #print(lams, mus)
            res = compute_probabilities(lams, mus)
            P[neogr].append(res)
            #print(res)

    Pijk = {}
    PisprTmp = {}
    for idxs in itertools.product(*map(lambda x:range(x+2), N1s)):
        vals = []
        for neogr in [False, True]:
            prod = 1
            for i,v in zip(idxs, P[neogr]):
                prod *= v[i]
            Pijk[(neogr,idxs)] = prod
    for idxs in itertools.product(*map(lambda x:range(x+2), N1s)):
        s,sno = (0,0)
        for idxs2 in itertools.product(*map(lambda x:range(x+1), idxs)):
            s += Pijk[(False,idxs2)]
            sno += Pijk[(True,idxs2)]
        PisprTmp[(False, idxs)] = s
        PisprTmp[(True, idxs)] = sno
        #print("P%d%d%d,%.10f,%.10f" % (idxs + (s,sno)) )

    Pispr = PisprTmp[(False, tuple(N1s))]
    PisprPrem = PisprTmp[(False, tuple(map(lambda x:x+1, N1s)))]
    Kg = Pispr / PisprPrem
    lotkaz = sum(l*n for l,n in zip(lambdas, Ns))
    mu_eq = list(map(lambda l:l["mu"][0], lambdamus[False]))
    mu_system = sum(mu_eq)
    Trem = 1 / mu_system
    Tno = (Kg * Trem) / (1 - Kg)
    return {
        "Kg" : Kg,
        "lambda_otkaz" : lotkaz,
        "mu_system" : mu_system,
        "Tno" : Tno,
        "Trem" : Trem,
        "P" : P,
        "lambdamus" : lambdamus
    }

def coctaianiia_csvlines(variant):
    def yield_csv(P, lambdamus):
        yield "i,nomer otkazavshih elementov i-ogo typa,veraiatnosth,lambda (i -> i+1),mu (i+1 -> i)\n"
        for i,probas in enumerate(P):
            for j,proba in enumerate(probas):
                l, m = (lambdamus[i][lm][j] if j<len(lambdamus[i][lm]) else 0 for lm in ("lambda","mu"))
                yield "%d,%d,%.10g,%.10g,%.10g\n" % (i+1, j, proba, l, m)
    res = delath_kursovuyu(variant)
    P = res["P"]
    lambdamus = res["lambdamus"]
    return {neogr:yield_csv(P[neogr],lambdamus[neogr]) for neogr in (False, True)}

def vlianie_params(variant):
    variations = {
        "lambdas": numpy.arange(0.00, 0.10, 0.01),
        "mu"     : numpy.arange(0.0 , 1.0, 0.1  )
    }
    for param in ("lambdas", "mu"):
        variant_mut = copy.deepcopy(variant)
        param_value = numpy.array(variant[param])
        for idx_modif in numpy.eye(len(param_value)):
            for delta in variations[param]:
                variant_mut[param] = param_value + idx_modif * delta
                yield (variant_mut, delath_kursovuyu(variant_mut))

def vlianie_csvlines(variant):
    res = list(vlianie_params(variant))
    headers_v = [ "%s_%d" % (header_name, header_idx+1)
                    for header_name in ("lambdas", "mu")
                    for header_idx in range(len(variant[header_name]))
                ]
    headers_res = None
    for var,res in vlianie_params(variant):
        if headers_res == None:
            headers_res = [k for k,v in res.items() if type(v) is numpy.float64]
            yield (",".join(headers_v + headers_res)) + "\n"
        yield (",".join([
                        str(var[h.split('_')[0]][int(h.split('_')[1])-1])
                            for h in headers_v
                        ] + [
                            "%.10g" % (res[x],) for x in headers_res
                        ]
               ) + "\n")

def make_csv(lines, filename):
    with open(filename, "w") as f:
        f.writelines(lines)

varianti = {
    "21" : {
        "lambdas" : [0.02, 0.07, 0.06],
        "mu"      : [0.7 , 1.3 , 1.2 ],
        "Ns"      : [7,    7,    4   ],
        "N1s"     : [4,    4,    2   ],
        "S"       : 2
    },
    "1" : {
        "lambdas" : [0.01, 0.05, 0.09],
        "mu"      : [1,    0.7,  1.5 ],
        "Ns"      : [3,    5,    3   ],
        "N1s"     : [2,    3,    2   ],
        "S"       : 2
    },
    "3" : {
        "lambdas" : [0.03, 0.07, 0.07],
        "mu"      : [1,    0.9,  1.5 ],
        "Ns"      : [5,    7,    4   ],
        "N1s"     : [3,    3,    2   ],
        "S"       : 2
    }
}

for nomer_variant,variant in varianti.items():
    j = os.path.join
    folder = j("resultati", "variant%s"%(nomer_variant,))
    os.makedirs(folder, exist_ok=True)
 
    make_csv(vlianie_csvlines(variant), j(folder, "Kg-Tno-Trem.csv"))
    probas = coctaianiia_csvlines(variant)
    for neogr in [False, True]:
        make_csv(probas[neogr],
                j(folder, "veraiatnosti_coctaianiia_%sogranicheni.csv" % ("ne" if neogr else "",)))
