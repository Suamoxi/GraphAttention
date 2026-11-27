"""Evaluation accepts only the frozen model, epoch and topology."""
import pytest
from omegaconf import OmegaConf

from scripts.evaluate_m49_generation import validate_checkpoint


@pytest.mark.parametrize("model,policy,epoch", [
    ("FullDiTGraphTransformer","keep",75),
    ("FullDiTGraphTransformer","keep",80),
    ("AlternatingDilatedGeometricDiT","keep",1000),
    ("AlternatingDilatedGeometricDiT","periodic_quotient",1000),
    ("AlternatingDilatedGeometricDiT","drop_max",25),
])
def test_m49_valid_checkpoint(model,policy,epoch):
    cfg = OmegaConf.create({"model":{"_target_":"some.module."+model},
                            "data":{"periodic_endpoint_policy":policy}})
    ckpt = {"epoch":epoch-1,"global_step":epoch*50,"model_class":model}
    assert validate_checkpoint(ckpt,cfg,epoch) == (model,policy)
    with pytest.raises(ValueError,match="epoch/update"):
        validate_checkpoint(ckpt,cfg,epoch+1)
    with pytest.raises(ValueError,match="frozen source"):
        cfg.model._target_ = "some.module.Wrong"
        validate_checkpoint(ckpt,cfg,epoch)


def test_m49_full_rejects_unqualified_periodic_topology_and_sampling_study():
    cfg = OmegaConf.create({"model":{"_target_":"some.module.FullDiTGraphTransformer"},
                            "data":{"periodic_endpoint_policy":"periodic_quotient"}})
    ckpt = {"epoch":79,"global_step":4000,"model_class":"FullDiTGraphTransformer"}
    with pytest.raises(ValueError,match="original stored-node"):
        validate_checkpoint(ckpt,cfg,80)
    cfg.data.periodic_endpoint_policy = "keep"
    with pytest.raises(ValueError,match="sampling study"):
        validate_checkpoint(ckpt,cfg,80,sampling_study=True)
