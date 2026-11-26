"""M47 relative-only, unique-node and periodic topology regression gates."""
from itertools import product
from types import SimpleNamespace

import pytest
import torch

from graph_attention.data.contracts import Mesh, Sample
from graph_attention.data.m47_variants import M47VariantDataset, periodic_edges_in_native_order
from graph_attention.geometry.relative import periodic_length_from_mesh_metadata
from graph_attention.geometry.connectivity import hex_connectivity_to_edge_index


def fixture(side=5):
    coords = torch.tensor(list(product(range(side), repeat=3)), dtype=torch.float64)
    offsets = ((0,0,0),(1,0,0),(1,1,0),(0,1,0),
               (0,0,1),(1,0,1),(1,1,1),(0,1,1))
    cells = torch.tensor([
        [((i+di)*side+j+dj)*side+k+dk for di,dj,dk in offsets]
        for i,j,k in product(range(side-1), repeat=3)
    ])
    order = torch.randperm(len(coords),generator=torch.Generator().manual_seed(42))
    inverse = torch.argsort(order)
    coords = coords[order]
    field = (coords.remainder(side-1)*torch.tensor([1.,7.,19.])).sum(1)
    mesh = Mesh(coords=coords,edge_index=torch.empty((2,0),dtype=torch.long),
                mesh_id="random_order",cell_connectivity=inverse[cells])
    def get(idx):
        return Sample(sample_id=f"Re54/Ma0.2:test_{idx}",mesh=mesh,
                      fields={"rho":field.clone()})
    src = SimpleNamespace(
        case_by_sample_id={f"Re54/Ma0.2:test_{i}":"Re54/Ma0.2" for i in range(2)},
        resolution_by_sample_id={f"Re54/Ma0.2:test_{i}":side for i in range(2)},
        case_definitions={"Re54/Ma0.2":SimpleNamespace(
            reference_scales={"L_ref":SimpleNamespace(value=2.0)})},
    )
    class Reader:
        def __len__(self):
            return 2
        def __getitem__(self,index):
            return get(index)
        def __getattr__(self,name):
            return getattr(src,name)
    return Reader(),mesh,field


def test_periodic_edges_with_permuted_exporter_indices():
    source, mesh, _ = fixture()
    unique = M47VariantDataset(source,variant="a3")[0].mesh
    edges = unique.edge_index
    n = 4**3
    assert edges.shape == (2,6*n)
    assert torch.equal(torch.bincount(edges[1],minlength=n),torch.full((n,),6))
    assert len(set(map(tuple,edges.T.tolist()))) == edges.shape[1]
    xyz = unique.coords
    diffs = xyz[edges[0]] - xyz[edges[1]]
    shortest = torch.remainder(diffs + 2.0, 4.0) - 2.0
    assert torch.equal(shortest.abs().sum(1),torch.ones(6*n,dtype=shortest.dtype))
    assert bool((diffs.abs()>1.0).any())  # actual across-face wrapping exists


def test_a1_a2_a3_node_counts_periodic_metadata_and_cache():
    source, mesh, field = fixture()
    a1 = M47VariantDataset(source,variant="a1")
    a2 = M47VariantDataset(source,variant="a2")
    a3 = M47VariantDataset(source,variant="a3")
    assert a1[0].mesh.num_nodes == 125
    assert a1[0].mesh is mesh
    open_sample = a2[0]
    assert open_sample.mesh.num_nodes == 64
    assert open_sample.mesh.num_edges == 0
    assert not open_sample.mesh.metadata["periodic_wrap_edges"]
    periodic = a3[0]
    assert periodic.mesh.num_nodes == 64
    assert periodic.mesh.num_edges == 384
    assert periodic.mesh.metadata["periodic_wrap_edges"]
    assert periodic.mesh.metadata["periodic_length_nondimensional"] == 2.0
    assert periodic_length_from_mesh_metadata((periodic.mesh.metadata,)) == 2.0
    assert periodic.mesh is a3[1].mesh
    assert a2[0].mesh is a2[1].mesh
    assert a2[0].mesh.num_nodes == 64
    assert hex_connectivity_to_edge_index(open_sample.mesh.cell_connectivity,64).shape[1] == 288


def test_inconsistent_duplicate_fails_closed():
    src, mesh, _ = fixture()
    original = src[0]
    broken = original.fields["rho"].clone()
    broken[(mesh.coords[:,0]==4).nonzero()[0]] += 1
    class BadReader:
        def __len__(self): return 1
        def __getitem__(self, index): return Sample(original.sample_id,mesh,{"rho":broken})
        def __getattr__(self,name): return getattr(src,name)
    with pytest.raises(ValueError,match="nonidentical periodic endpoint"):
        M47VariantDataset(BadReader(),variant="a2")[0]


def test_mesh_metadata_periodic_length_contract():
    assert periodic_length_from_mesh_metadata(({},)) is None
    assert periodic_length_from_mesh_metadata(({},{})) is None
    with pytest.raises(ValueError,match="mixed periodic"):
        periodic_length_from_mesh_metadata(({}, {"periodic_length_nondimensional":1.0}))
    with pytest.raises(ValueError,match="unequal"):
        periodic_length_from_mesh_metadata(({"periodic_length_nondimensional":1.0},
                                           {"periodic_length_nondimensional":2.0}))
    assert periodic_length_from_mesh_metadata((
        {"periodic_length_nondimensional":2.0},
        {"periodic_length_nondimensional":2.0})) == 2.0


def test_invalid_variant():
    with pytest.raises(ValueError,match="variant"):
        M47VariantDataset(object(),variant="a0")


def test_periodic_edge_guard_rejects_noncartesian():
    src, mesh, _ = fixture()
    coords = M47VariantDataset(src,variant="a2")[0].mesh.coords.clone()
    coords[0,0] += .15
    with pytest.raises(ValueError,match="unique Cartesian lattice"):
        periodic_edges_in_native_order(coords,side=4)
