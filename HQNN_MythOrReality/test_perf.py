import time
import torch
from merlin import build_slos_distribution_computegraph as build_slos_graph
from itertools import combinations
import numpy as np


def generate_all_fock_states(m, n, no_bunching=False):
    """Generates all possible Fock states for m modes and n photons."""
    if no_bunching:
        if n > m or n < 0:
            return
        for positions in combinations(range(m), n):
            fock_state = [0] * m

            for pos in positions:
                fock_state[pos] = 1
            yield tuple(fock_state)

    else:
        if n == 0:
            yield (0,) * m
            return
        if m == 1:
            yield (n,)
            return

        for i in reversed(range(n + 1)):
            for state in generate_all_fock_states(m - 1, n - i):
                yield (i,) + state


m = 16
n = 8

iteration = 5
nb = [True, False]
graph_time = []
retrieval_time = []
generate_time = []

for _ in range(iteration):
    for no_bunching in nb:
        print(f"\n ----- No bunching = {no_bunching} ----- \n")
        start_time = time.time()
        slos_graph = build_slos_graph(m, n, keep_keys=True, no_bunching=no_bunching)
        mid_time = time.time()
        keys1, _ = slos_graph.compute(torch.eye(m, dtype=torch.complex64), [1] * n + [0] * (m - n))
        end_time = time.time()
        print('Total time with compute:', end_time - start_time)
        print(f' - Graph building', mid_time - start_time)
        print(f' - Graph compute', end_time - mid_time)
        graph_time.append(end_time - mid_time)

        start_time = time.time()
        slos_graph = build_slos_graph(m, n, keep_keys=True, no_bunching=no_bunching)
        mid_time = time.time()
        keys1_bis = slos_graph.final_keys
        end_time = time.time()
        print('\nTotal time with init:', end_time - start_time)
        print(f' - Graph building', mid_time - start_time)
        print(f' - Get keys from graph', end_time - mid_time)
        retrieval_time.append(end_time - mid_time)

        start_time = time.time()
        keys2 = list(generate_all_fock_states(m, n, no_bunching=no_bunching))
        print('\nTotal time with generate_all_fock_state :', time.time() - start_time)
        generate_time.append(time.time() - start_time)
        #print(f"Keys from compute: {list(keys1)}")
        #print(f"Keys from init: {list(keys1_bis)}")
        #print(f"Keys from generate_all_fock_states: {list(keys2)}")

        if keys1 == keys2 and keys1_bis == keys1:
            print('All is good!')
        else:
            print('All is bad!')

print(f"\n ---- Summary after {iteration} iterations ---- \n")
print(f"Average graph compute time: {np.mean(graph_time)} seconds")
print(f"Average graph retrieval time: {np.mean(retrieval_time)} seconds")
print(f"Average generate_all_fock_states time: {np.mean(generate_time)} seconds")
best = "Generation of all Fock states" if np.mean(generate_time) < min(np.mean(graph_time), np.mean(retrieval_time)) else ("Retrieval of keys from graph" if np.mean(retrieval_time) < np.mean(graph_time) else "Graph compute")
print(f"Fastest way on {iteration} iterations is {best} ")