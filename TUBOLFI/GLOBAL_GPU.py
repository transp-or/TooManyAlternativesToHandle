# This is a py file to store all global variables used across modules
import numpy as np
from typing import Callable
import numpy.typing as npt
from typing import Optional
import torch
"""Shared runtime state, populated by the runner.

REAL_BETA / REAL_BETA_U: true and estimable coefficient vectors.
REAL_BETA_dict: ordered coefficient names and values.
size, J, P: observation, alternative and feature counts.
ATTRIBUTE_MATRIX: feature tensor (size, J, P).
SCALED_ATTRIBUTE_MATRIX: discrepancy features; currently an unscaled copy.
OBSERVATION_MATRIX: observed alternative indices (size,), not one-hot choices.
EPSILION: numerical floor; MASTER_SEED: base RNG seed.
DEVICE and DATA_TYPE_*: tensor device and numerical precision.
"""


REAL_BETA: Optional[torch.Tensor] =  None
REAL_BETA_dict: Optional[dict] =  None
REAL_BETA_U: Optional[torch.Tensor] =  None
J: Optional[int] =  None
size: Optional[int] =  None
P: Optional[int] =  None
ATTRIBUTE_MATRIX: Optional[torch.Tensor] =  None
SCALED_ATTRIBUTE_MATRIX: Optional[torch.Tensor] =  None
OBSERVATION_MATRIX: Optional[torch.Tensor] =  None
DATA_TYPE_torch =torch.float64#= None #torch.float64
DATA_TYPE_np = np.float64#= None #np.float64
EPSILION :Optional[float] = None
MASTER_SEED = 42
DEVICE :str = 'cpu'
SIMULATOR_TYPE: str = "all"   # case-insensitive: "all" (utility+epsilon argmax) or "SA"/"sa" (sequential/MH-based discrete choice simulator)

