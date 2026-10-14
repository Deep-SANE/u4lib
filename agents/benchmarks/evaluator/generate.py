"""Freeze deterministic SU(4) integration fixtures (tasks 4 and 5) before runs.

Reuses the library-agnostic u3lib grade.py as the grader; only the expected
values differ, and are computed here with u4lib on the host.
"""
import itertools as it, json, sys, hashlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]           # QCX
sys.path.insert(0, str(ROOT / 'u4lib/python'))
sys.path.insert(0, str(ROOT / 'u3lib/python'))
import u4lib as u

def irr(x): return u.irrep(*x)
def eta(a, b, c): return u.Eta(irr(a), irr(b), irr(c))

def channels(a, b, c, t, left):
    """Intermediates X with eta(a,b,X)*eta(X,c,t) (left) or eta(a,b,X)*eta(c,X,t)."""
    out = []
    for X in it.product(range(6), repeat=3):          # P1,P2,P3 in 0..5
        r = eta(a, b, X)
        s = eta(X, c, t) if left else eta(c, X, t)
        if r * s:
            out += [list(X) + [i, j] for i in range(r) for j in range(s)]
    return out

def gen_task4():
    cases = []
    candidates = [
        ('U', (1,0,0), (1,0,0), (1,0,0), (1,0,0)),
        ('Z', (1,0,0), (1,0,0), (1,0,0), (1,0,0)),
        ('U', (1,0,1), (1,0,0), (1,0,0), (1,0,1)),
        ('Z', (1,0,1), (1,0,0), (1,0,0), (1,0,1)),
        ('U', (1,0,1), (1,0,1), (1,0,1), (1,0,1)),
        ('Z', (1,0,1), (1,0,1), (1,0,1), (1,0,1)),
    ]
    for kind, A, B, C, T in candidates:
        left = channels(A, B, C, T, True)
        right = channels(B, C, A, T, False) if kind == 'U' else channels(A, C, B, T, True)
        if not left or len(left) != len(right):
            continue
        v = np.zeros((len(left), len(right)))
        AI, BI, CI, TI = map(irr, (A, B, C, T))
        for i, (*xl, r, s) in enumerate(left):
            X = tuple(xl)
            for j, (*yl, p, q) in enumerate(right):
                Y = tuple(yl)
                if kind == 'U':
                    o = u.u4u6db().get_U4U6(AI, BI, TI, CI, irr(X), eta(A,B,X), eta(X,C,T),
                                            irr(Y), eta(B,C,Y), eta(A,Y,T))
                else:
                    o = u.u4z6db().get_U4Z6(BI, AI, TI, CI, irr(X), eta(A,B,X), eta(X,C,T),
                                            irr(Y), eta(A,C,Y), eta(Y,B,T))
                v[i, j] = o(r, s, p, q)
        if np.linalg.norm(v @ v.T - np.eye(len(left))) > 1e-9:
            continue
        cases.append(dict(request=[kind, list(A), list(B), list(C), list(T)],
                          left=left, right=right, value=v.tolist()))
    assert cases, 'no task-4 cases generated'
    return cases

def gen_task5():
    cases = []
    pool = [(1,0,0),(0,1,0),(0,0,1),(1,0,1),(2,0,0),(0,0,2)]
    quad = [(1,0,0),(1,0,0),(1,0,0),(1,0,0)]
    A,B,C,D = quad
    # search intermediate/final combos with all six etas positive
    seen = 0
    for AB,CD,AC,BD in it.product(pool, repeat=4):
        for T in pool:
            rs = [eta(A,B,AB), eta(C,D,CD), eta(AB,CD,T), eta(A,C,AC), eta(B,D,BD), eta(AC,BD,T)]
            if not all(rs):
                continue
            req = [A,B,C,D,AB,CD,AC,BD,T]
            o = u.u4u9db().get_U4U9(*map(irr,(A,B,C,D,AB,CD,AC,BD,T)),
                                    rs[3], rs[4], rs[2], rs[0], rs[1], rs[5])
            entries = []
            for a,b,l,c,e,r in it.product(*(range(x) for x in rs)):
                entries.append([a,b,l,c,e,r, float(o(c,e,l,a,b,r))])
            cases.append(dict(request=[list(x) for x in req], shape=rs, entries=entries))
            seen += 1
            break
        if seen >= 3:
            break
    # also include one adjoint request for a multiplicity>1 block if available
    adj=(1,0,1)
    rs=[eta(adj,adj,X) for X in [adj]]  # placeholder; real search below
    for AB in pool:
        rs=[eta(adj,adj,AB), eta(adj,adj,AB), eta(AB,AB,adj), eta(adj,adj,AB), eta(adj,adj,AB), eta(AB,AB,adj)]
        if all(rs) and max(rs)>1:
            req=[adj,adj,adj,adj,AB,AB,AB,AB,adj]
            o=u.u4u9db().get_U4U9(*map(irr,req), rs[3],rs[4],rs[2],rs[0],rs[1],rs[5])
            entries=[[a,b,l,c,e,r,float(o(c,e,l,a,b,r))] for a,b,l,c,e,r in it.product(*(range(x) for x in rs))]
            cases.append(dict(request=[list(x) for x in req], shape=rs, entries=entries))
            break
    assert cases, 'no task-5 cases generated'
    return cases

def gen_task6():
    cases = []
    raw_requests = [
        ('C', [1, 0, 0, 0, 1, 0, 0,  0, 0, 0, 0, 0, 0, 0,  1, 0, 0, 0, 1, 0, 0]),
        ('P', [1, 0, 0, 1, 1,  0, 0, 1, 1, 1,  1, 0, 1, 2, 2]),
        ('C', [3, 2, 1, 0, 3, 2, 1,  3, 2, 1, 0, 3, 2, 0,  5, 4, 2, 1, 5, 4, 2]),
        ('P', [1, 0, 0, 1, 1,  0, 0, 1, 1, 1,  1, 0, 1, 2, 0]),
        ('C', [1, 0, 0, 0, 1, 0, 0,  1, 0, 0, 0, 1, 0, 0,  2, 0, 0, 0, 2, 0, 0]),
        ('P', [1, 0, 1, 2, 2,  1, 1, 1, 2, 2,  1, 1, 1, 2, 2]),
    ]
    for tag, labels in raw_requests:
        req_str = f"{tag} " + " ".join(map(str, labels))
        if tag == 'C':
            arr = u.wigU4U3db().get(*labels)
            rhomax, etamax = arr.shape
            entries = []
            for r in range(rhomax):
                for e in range(etamax):
                    entries.append([r, e, float(arr[r, e])])
            cases.append(dict(tag='canon', request=req_str, shape=[rhomax, etamax], entries=entries))
        else:
            obj = u.wigU4SU2SU2db().get_U4SU2SU2(*labels)
            shape = [obj.zetaAmax(), obj.zetaBmax(), obj.zetaCmax(), obj.etamax()]
            entries = []
            for za, zb, zc, e in it.product(*(range(x) for x in shape)):
                entries.append([za, zb, zc, e, float(obj(za, zb, zc, e))])
            cases.append(dict(tag='phys', request=req_str, shape=shape, entries=entries))
    return cases

if __name__ == '__main__':
    u.initialize()
    try:
        data = {'4': gen_task4(), '5': gen_task5(), '6': gen_task6()}
    finally:
        u.finalize()
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / 'fixtures-pilot.json'
    target.write_text(json.dumps(data, indent=2) + '\n')
    print({k: len(v) for k, v in data.items()})
    print('task5 shapes:', [c['shape'] for c in data['5']])
    print('task4 sizes:', [(c['request'][0], len(c['left'])) for c in data['4']])
    print('sha256', hashlib.sha256(target.read_bytes()).hexdigest())
