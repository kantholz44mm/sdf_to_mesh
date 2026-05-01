import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from interpreter import Interpreter, type_of, SDFTypeError, SDFNameError

@pytest.fixture(scope='session')
def interp():
    return Interpreter()

def run(interp, src):
    return interp.run_str(src)

def approx(val, rel=1e-5):
    return pytest.approx(val, rel=rel)

def eval_sdf(fn, *points):
    """Evaluate an SDF callable at one or more (x,y,z) points, return list of floats."""
    pts = np.array(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    out = fn(pts)
    return [float(v) for v in np.atleast_1d(out.squeeze())]
