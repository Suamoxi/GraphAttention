"""Analytic checks of physical quadrature and unpaired global statistics."""
import numpy as np
import pytest

from scripts.audit_m49_global_integrals import (
    native_quadrature, snapshot_record, population, compare,
)
from graph_attention.data.periodic_endpoints import reduce_cartesian_endpoints
from test_m49_periodic_endpoints import mesh_and_field


def test_native_periodic_quadrature():
    mesh, field = mesh_and_field(side=9)
    reduction = reduce_cartesian_endpoints(mesh,side=9,policy="periodic_quotient")
    volume,weights = native_quadrature(mesh,reduction)
    assert volume == 8**3
    assert weights.sum() == 1
    assert weights.min() == 1/(8*volume)
    assert weights.max() == 1/volume
    assert weights @ field.numpy() == pytest.approx(field[reduction.retained].mean().item())


def test_global_integrals_units_and_endpoint_ambiguity():
    q = np.tile([2., 3., -4., 5., 100.],(8,1))
    scales = np.array([1.5, 6., 6., 6., 24.])
    volume = 2.
    full = np.concatenate((q, q[:1]+np.array([9.,0,0,0,0])),axis=0)
    record = snapshot_record(q,"a",scales=scales,volume=volume,
                             full=full,weights=np.full(len(full),1/len(full)))
    np.testing.assert_allclose(record["integrals_si"],[6,36,-48,60,4800])
    np.testing.assert_allclose(record["bulk_velocity_m_per_s"],[6,-8,10])
    assert record["thermodynamic_invalid_fraction"] == 0
    np.testing.assert_allclose(record["native_minus_unique_mean_nondimensional"],
                               [1,0,0,0,0],atol=1e-14)
    altered = q.copy()
    altered[:,0] += .25
    pop = population([record])
    gen = population([snapshot_record(altered,"b",scales=scales,volume=volume)])
    result = compare(pop,gen,scales,volume,np.ones(5))
    assert result["rho"]["mean_bias_integral_si"] == pytest.approx(.75)
    assert result["rho"]["wasserstein_integral_si"] == pytest.approx(.75)
    assert result["rho"]["reference_std_nondimensional"] == 0
    assert result["rhou"]["wasserstein_integral_si"] == 0


def test_invalid_state_not_clipped():
    q = np.tile([1.,0.,0.,0.,1.],(8,1))
    q[0,0] = -1.
    record = snapshot_record(q,"bad",scales=np.ones(5),volume=1.)
    assert record["nonpositive_density_fraction"] == .125
    assert record["thermodynamic_invalid_fraction"] == .125
    assert record["integrals_si"][0] == .75
    q[1,0] = np.nan
    record = snapshot_record(q,"nan",scales=np.ones(5),volume=1.)
    assert record["finite_unique"] is False
    assert "integrals_si" not in record
