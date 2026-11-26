from itertools import product

import pytest
import torch

from graph_attention.data.contracts import Mesh, Sample
from graph_attention.data.periodic_endpoints import (
    lift_periodic_endpoints, reduce_cartesian_endpoints, reduce_sample,
)
from graph_attention.geometry.connectivity import hex_connectivity_to_edge_index


def mesh_and_field(side=5):
    coords = torch.tensor(list(product(range(side), repeat=3)), dtype=torch.float64)
    local = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),
             (0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    cells = torch.tensor([
        [((i+x)*side+j+y)*side+k+z for x,y,z in local]
        for i,j,k in product(range(side-1),repeat=3)
    ])
    # Deliberately non-Cartesian exporter order.
    perm = torch.randperm(len(coords), generator=torch.Generator().manual_seed(42))
    inverse = torch.argsort(perm)
    coords = coords[perm]
    field = (coords.remainder(side-1)*torch.tensor([1.,7.,19.])).sum(1)
    return Mesh(coords,torch.empty((2,0),dtype=torch.long),
                mesh_id="test",cell_connectivity=inverse[cells]), field


def test_periodic_endpoint_roundtrip_and_induced_open_graph():
    mesh, field = mesh_and_field()
    reduction = reduce_cartesian_endpoints(mesh,side=5)
    reduced = reduce_sample(Sample("test",mesh,{"rho":field}),reduction)
    assert reduced.mesh.num_nodes == 64
    assert reduced.mesh.cell_connectivity.shape == (27,8)
    torch.testing.assert_close(lift_periodic_endpoints(reduced.fields["rho"][:,None],reduction),
                               field[:,None],rtol=0,atol=0)
    edge=hex_connectivity_to_edge_index(reduced.mesh.cell_connectivity,64)
    original_edge=hex_connectivity_to_edge_index(mesh.cell_connectivity,125)
    old_to_new=torch.full((125,),-1,dtype=torch.long)
    old_to_new[reduction.retained]=torch.arange(64)
    induced=old_to_new[original_edge]
    induced=induced[:,(induced>=0).all(0)]
    assert set(map(tuple,edge.T.tolist()))==set(map(tuple,induced.T.tolist()))
    assert edge.shape[1]==2*3*3*4**2
    distance=(reduced.mesh.coords[edge[0]]-reduced.mesh.coords[edge[1]]).abs().sum(1)
    assert torch.equal(distance,torch.ones_like(distance))


def test_nonidentical_endpoint_rejected():
    mesh, field=mesh_and_field()
    reduction=reduce_cartesian_endpoints(mesh,side=5)
    field[torch.nonzero((mesh.coords==4).any(1),as_tuple=True)[0][0]] += 1
    with pytest.raises(ValueError,match="nonidentical periodic endpoint"):
        reduce_sample(Sample("bad",mesh,{"rho":field}),reduction)


def test_noncartesian_mesh_rejected():
    mesh,_=mesh_and_field()
    mesh.coords[0,0] += .1
    with pytest.raises(ValueError,match="uniform Cartesian"):
        reduce_cartesian_endpoints(mesh,side=5)
