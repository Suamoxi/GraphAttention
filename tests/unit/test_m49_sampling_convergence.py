import pytest
import torch
from scripts.compare_m49_sampling import matched_errors


def test_matched_errors_exclude_duplicated_nodes_and_report_zero_norm():
    a=torch.tensor([[[2.,1.],[100.,100.],[1.,1.]]])
    b=torch.tensor([[[4.,0.],[-100.,-100.],[2.,0.]]])
    result=matched_errors(a,b,torch.tensor([True,False,True]),("rho","momentum"))
    assert result["rho"]["relative_l2_mean"]==pytest.approx(.5)
    assert result["rho"]["standardized_rmse_mean"]==pytest.approx((2.5)**.5)
    assert result["momentum"]["relative_l2_mean"] is None
    assert result["momentum"]["standardized_rmse_mean"]==1
