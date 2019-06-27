import nltk
import numpy as np
import six
from nltk.parse.generate import generate
import pdb
gram = """C -> S 'and' S
    C -> S 'after' S
    C -> S
    S -> V 'twice'
    S -> V 'thrice'
    S -> V
    V -> D
    V -> U
    V -> 'turn' 'opposite' 'right' 
    V -> 'turn' 'opposite' 'left'
    V -> 'walk' 'opposite' 'right' 
    V -> 'walk' 'opposite' 'left'
    V -> 'run' 'opposite' 'right' 
    V -> 'run' 'opposite' 'left'
    V -> 'jump' 'opposite' 'right' 
    V -> 'jump' 'opposite' 'left'
    V -> 'look' 'opposite' 'right' 
    V -> 'look' 'opposite' 'left'
    V -> 'turn' 'around' 'right' 
    V -> 'turn' 'around' 'left'
    V -> 'walk' 'around' 'right' 
    V -> 'walk' 'around' 'left'
    V -> 'run' 'around' 'right' 
    V -> 'run' 'around' 'left'
    V -> 'jump' 'around' 'right' 
    V -> 'jump' 'around' 'left'
    V -> 'look' 'around' 'right' 
    V -> 'look' 'around' 'left'
    D -> U 'left'
    D -> U 'right'
    D -> 'turn' 'left'
    D -> 'turn' 'right'
    U -> 'walk'
    U -> 'look'
    U -> 'run'
    U -> 'jump' 
    Nothing -> None
    """

GCFG = nltk.CFG.fromstring(gram)
start_index = GCFG.productions()[0].lhs()

all_lhs = [a.lhs().symbol() for a in GCFG.productions()]
lhs_list = ["<pad>", "<EOS>"]
for a in all_lhs:
    if a not in lhs_list:
        lhs_list.append(a)

D = len(GCFG.productions())

rhs_map = [None]*(D+2)
count = 2
for a in GCFG.productions():
    rhs_map[count] = []
    for b in a.rhs():
        if not isinstance(b,six.string_types):
            s = b.symbol()
            rhs_map[count].extend(list(np.where(np.array(lhs_list) == s)[0]))
    count = count + 1

masks = np.zeros((len(lhs_list), D+2))
count = 0
for idx, sym in enumerate(lhs_list):
    if sym == "<pad>" or sym == "<EOS>":
        sym = "Nothing"
    is_in = np.array([0, 0] + [a == sym for a in all_lhs], dtype=int).reshape(1,-1)
    masks[count] = is_in
    count = count + 1

index_array = []
for i in range(masks.shape[1]):
    try:
        index_array.append(np.where(masks[:,i]==1)[0][0])
    except:
        index_array.append(0)
ind_of_ind = np.array(index_array)

# CFL
CFL = []
for sentence in generate(GCFG):
    CFL.append(" ".join(sentence))

# production_rule -> id
prod_map = {"<pad>":0, "<EOS>":1}
for ix, prod in enumerate(GCFG.productions()):
    prod_map[prod] = ix+2

# parser
parser = nltk.ChartParser(GCFG)

# U=['walk', 'look', 'run', 'jump')]
# D=[u+" left" for u in U]
# D+=[u+" right" for u in U]
# D+=['turn left', 'turn right']
# V=D+U
# V+=[d.split()[0]+" opposite "+d.split()[1] for d in D]
# V+=[d.split()[0]+" around "+d.split()[1] for d in D]
# S=copy.deepcopy(V)
# S+=[v + " twice" for v in V]
# S+=[v + " thrice" for v in V]
# C=copy.deepcopy(S)
# for s in S:
#     for s2 in S:
#         C+=[s + " and " + s2]
#         C+=[s + " after " + s2]
# C=[" ".join(c.split()) for c in C]

