import h5py
import numpy as np
import pytest

from scripts.qualify_m49_hit_archive import read_primitives, diagnostics, select_positions


def test_named_primitive_fields_and_eos(tmp_path):
    path=tmp_path/"data.h5"
    rho=np.ones((4,4,4))
    v=np.zeros((*rho.shape,3))
    v[...,0]=np.arange(4)[:,None,None]
    with h5py.File(path,"w") as h:
        h["primes/density"]=rho
        h["primes/pressure"]=2*rho
        h["primes/temperature"]=rho
        h["primes/velocity"]=np.moveaxis(v,-1,0)
        r,p,t,u,mapping=read_primitives(h)
        np.testing.assert_array_equal(u,v)
        assert mapping["velocity_component_axis"]==0
        setup={"HIT.json":{"content":{
            "material_properties":{"type":"IdealGas","specific_heat_ratio":1.4,
                                   "specific_gas_constant":2.0,"dynamic_viscosity":"lambda T: 0.01 * T**0.76"},
            "domain":{a:{"cells":4,"range":[0.,2.]} for a in "xyz"}}}}
        result=diagnostics(r,p,t,u,setup)
        assert result["finite"]
        assert result["eos_relative_l2_temperature_error"]==0
        assert result["fluctuation_tke"]==pytest.approx(.625)
        assert result["exploratory_Re_box"]>0
        h["rho"]=rho
        with pytest.raises(ValueError,match="ambiguous aliases"):
            read_primitives(h)


def test_unknown_schema_is_not_inferred_from_shape(tmp_path):
    with h5py.File(tmp_path/"data.h5","w") as h:
        h["anonymous"]=np.ones((32,32,32))
        with pytest.raises(ValueError,match="explicitly named"):
            read_primitives(h)
    indices=select_positions(30000,33)
    assert indices[:8]==list(range(8))
    assert indices[-8:]==list(range(29992,30000))
