
from typing import NamedTuple, Optional, List, Any

import numpy as np

START = np.ones((1, 768)) * 10

class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack

class Alignment(NamedTuple):
    alignment: np.array
    consensus: Optional[np.array]
    gaps: Optional[np.array]
    dists: Optional[np.array]
    ys: np.array
    cols: np.array


class AlignmentInfo(NamedTuple):
    # algo keys
    algo: str
    subalgo: Optional[str]
    G: float
    Gx: float
    Gz: float
    Gxz: float

    # alignment keys
    alignment: np.array
    ys: np.array
    consensus: Optional[np.array]
    gaps: Optional[np.array]
    cols: Optional[np.array]

    # distance keys
    steiner_dists: np.array
    sp_dists: Optional[np.array]
    star_dists: Optional[np.array]
    pairwise_dists: Optional[np.array]

    # multiple alignment
    var: np.array
    sse: np.array

    # meta info
    cluster: np.array
    score_history: List[float]
