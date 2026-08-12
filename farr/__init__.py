"""Full FARR: forward-looking retrieval with answer verification and revision."""

from .config import FARRConfig
from .pipeline import FARR as FARRV1
from .pipeline_v2 import FARRV2
from .pipeline_v3 import FARRV3
from .types import FARRResult, FARRStats, HopTrace, VerificationTrace

FARR = FARRV3

__all__ = [
    "FARR",
    "FARRV1",
    "FARRV2",
    "FARRV3",
    "FARRConfig",
    "FARRResult",
    "FARRStats",
    "HopTrace",
    "VerificationTrace",
]
