"""Shortest-path shells, including directed graphs and disconnected packs."""
from collections import deque
from itertools import product

import torch
from graph_attention.geometry import build_attention_edge_indices, exact_three_hop_edge_index


def edge_set(edges):
    return set(map(tuple,edges.T.tolist()))


def bfs_shell(edges,n,hop):
    adjacency=[set() for _ in range(n)]
    for i,j in edges.T.tolist():
        adjacency[i].add(j)
    result=set()
    for i in range(n):
        dist={i:0}
        queue=deque([i])
        while queue:
            a=queue.popleft()
            if dist[a]==hop:
                continue
            for b in adjacency[a]:
                if b not in dist:
                    dist[b]=dist[a]+1
                    queue.append(b)
        result.update((i,j) for j,d in dist.items() if d==hop)
    return result


def test_exact_three_directed_disconnected_triangles_and_duplicates():
    e=torch.tensor([(0,1),(0,1),(1,2),(2,0),(2,3),(3,4),(0,3),
                    (5,6),(6,7),(7,8),(8,5),(8,8)],dtype=torch.long).T.contiguous()
    top=build_attention_edge_indices(e,10,{"dilated":"exact_two_hop","dilated_3":"exact_three_hop"})
    for hop,name in ((2,"dilated"),(3,"dilated_3")):
        assert edge_set(top[name])==bfs_shell(e,10,hop)
        assert len(edge_set(top[name]))==top[name].shape[1]
    assert torch.equal(top["dilated_3"],exact_three_hop_edge_index(e,10))
    perm=torch.randperm(10,generator=torch.Generator().manual_seed(7))
    assert edge_set(exact_three_hop_edge_index(perm[e],10))==edge_set(perm[top["dilated_3"]])
    assert exact_three_hop_edge_index(torch.empty((2,0),dtype=torch.long),10).shape==(2,0)


def test_cartesian_shell_degree():
    side=7
    coords=list(product(range(side),repeat=3))
    ids={c:i for i,c in enumerate(coords)}
    pairs=[]
    for i,c in enumerate(coords):
        for a in range(3):
            for d in (-1,1):
                q=list(c);q[a]+=d
                if tuple(q) in ids:
                    pairs.append((i,ids[tuple(q)]))
    edges=torch.tensor(pairs).T.contiguous()
    three=exact_three_hop_edge_index(edges,len(coords))
    center=ids[(3,3,3)]
    assert int((three[1]==center).sum())==38
    assert edge_set(three)==bfs_shell(edges,len(coords),3)
