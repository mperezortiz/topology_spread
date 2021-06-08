from itertools import product
from multiprocessing import Pool
from experiment import run_experiment

# this script uses parallelization

farz_runs = []

# k, b, r, q, a, g
def create_farz(**kwargs):
    conf_dict = {"network_family": 'farz'}
    conf_dict.update(kwargs)
    return conf_dict

# main parameters for farz
#     n: number of nodes
#     m: number of edges created per node (i.e. half the average degree of nodes)
#     k: number of communities
# control parameters
#     beta: the strength of community structure, i.e. the probability of edges to be formed within communities, default (0.8)
#     alpha: the strength of common neighbor's effect on edge formation edges, default (0.5)
#     gamma: the strength of degree similarity effect on edge formation, default (0.5), can be negative for networks with negative degree correlation
# overlap parameters
#     r: the maximum number of communities each node can belong to, default (1, which results in disjoint communities)
#     q: the probability of a node belonging to the multiple communities, default (0.5, has an effect only if r>1)
# config parameters
#     phi: the constant added to all community sizes, higher number makes the communities more balanced in size, default (1) which results in power law distribution for community sizes
#     epsilon: the probability of noisy/random edges, default (0.0000001)
#     t: the probability of also connecting to the neighbors of a node each nodes connects to. The default value is (0), but could be increase to a small number to achieve higher clustering coefficient.

params = [[5, 10, 50, 100], [0.3, 0.5, 0.8], [0.25, 0.33, 0.5],
          [0.1, 0.3, 0.5, 0.7], [0.0, 0.2, 0.5, 0.8],
          [-0.3, -0.1, 0.2, 0.5, 0.8]]

for k, b, r, q, a, g, nodes, edges in product(*params):
    farz_runs.append(
        create_farz(k=k, b=b, r=r, q=q, a=a, g=g, nodes=nodes, edges=edges))

with Pool(20) as p:
    p.starmap(run_experiment, product(farz_runs, range(1)))

