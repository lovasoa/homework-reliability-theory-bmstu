import numpy
import itertools
import copy
import sys
from numpy.linalg import solve


def delath_kursovuyu(variant):
    lambdas = variant["lambdas"]
    mu      = variant["mu"]
    Ns      = variant["Ns"]
    N1s     = variant["N1s"]

    P = {True:[], False:[]}
    lambdamus = {True:[], False:[]}

    def get_lams(step, neogr):
        if not neogr:
            return [i * lambdas[step] for i in range(Ns[step], Ns[step]-N1s[step]-1, -1)]
        else:
            return [i * lambdas[step] for i in range(Ns[step], 0, -1)]

    def get_mus(step, neogr):
        N = Ns[step]
        N1 = N1s[step]+1
        if neogr:
            if step >=1:
                PP = P[neogr][step-1]
                P0, P1 = PP[0,0], PP[1,0]
                return [(1+i)*mu[step]*P0 + mu[step]*P1 for i in range(N)]
            else:
                return [(1+i)*mu[step] for i in range(N)]
        else:
            if step==0: mu1, mu2 = (mu[step], 2*mu[step])
            else:
                PP = P[neogr][step-1]
                mP0, mP1 = mu[step]*PP[0,0], mu[step]*PP[1,0]
                mu1, mu2 = (mP0+mP1, 2*mP0+mP1)
            return [(mu1 if i==0 else mu2) for i in range(N1)]

    def compute_probabilities(p_lambdas, p_mus):
        N = len(p_lambdas)+1
        lams = {k:i for k,i in enumerate(p_lambdas)}
        mus  = {k:i for k,i in enumerate(p_mus)}
        
        def coeff(i,j, lams, mus):
            if i == j:
                return -(lams.get(j, 0) + mus.get(j-1,0))
            if i == j+1:
                return mus.get(j,0)
            if i == j-1:
                return lams.get(i,0)
            else: return 0

        mat = numpy.mat([
            [coeff(i,j, lams, mus) for i in range(N)]
            for j in range(N)
        ])

        eigenvals, eigenvects = numpy.linalg.eig(mat)
        idx = eigenvals.argsort()[::-1][0]
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
                prod *= v[i][0,0]
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
        yield "i, nomer otkazavshih elementov i-ogo typa, veraiatnosth, lambda (i -> i+1), mu (i+1 -> i)\n"
        for i,probas in enumerate(P):
            for j,proba in enumerate(probas):
                l, m = (lambdamus[i][lm][j] if j<len(lambdamus[i][lm]) else 0 for lm in ("lambda","mu"))
                yield "%d, %d, %.10g, %.10g, %.10g\n" % (i+1, j, proba, l, m)
    res = delath_kursovuyu(variant)
    P = res["P"]
    lambdamus = res["lambdamus"]
    return {neogr:yield_csv(P[neogr],lambdamus[neogr]) for neogr in (False, True)}

def vlianie_params(variant):
    variations = {
        "lambdas": numpy.arange(0.01, 0.11, 0.01),
        "mu"     : numpy.arange(0.5 , 1.5, 0.1 )
    }
    for param in ("lambdas", "mu"):
        variant_mut = copy.deepcopy(variant)
        param_value = numpy.array(variant[param])
        for i in numpy.eye(len(param_value)):
            for j in variations[param]:
                variant_mut[param] = (param_value - i*param_value) + i*j
                yield (variant_mut, delath_kursovuyu(variant_mut))

def vlianie_csvlines(variant):
    res = list(vlianie_params(variant))
    headers_v = [ str(k) + '_' + str(n+1)
                    for k,v in variant.items()
                    for n in range(len(v))
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
        "N1s"     : [4,    4,    2   ]
    },
    "1" : {
        "lambdas" : [0.01, 0.05, 0.09],
        "mu"      : [1,    0.7,  1.5 ],
        "Ns"      : [3,    5,    3   ],
        "N1s"     : [2,    3,    2   ]
    },
    "3" : {
        "lambdas" : [0.03, 0.07, 0.07],
        "mu"      : [1,    0.9,  1.5 ],
        "Ns"      : [5,    7,    4   ],
        "N1s"     : [3,    3,    2   ]
    }
}

if len(sys.argv) == 2:
    variant = varianti.get(sys.argv[1])
    if variant == None:
        print("Variant '%s' doen't exist." % (sys.argv[1],), file=sys.stderr)
        sys.exit(2)
else:
    print("Usage: %s VARIANT.\nPlease specify one of the variants defined in the code." % (sys.argv[0],), file=sys.stderr)
    sys.exit(1)

make_csv(vlianie_csvlines(variant), "Kg-Tno-Trem.csv")
probas = coctaianiia_csvlines(variant)
for neogr in [False, True]:
    make_csv(probas[neogr],
            "veraiatnosti_coctaianiia_%sogranicheni.csv" % ("ne" if neogr else "",))
