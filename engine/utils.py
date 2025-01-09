import numpy as np
from engine.autograd import Scalar

def toOneHot(data):
  res = []
  for d in data:
    one_hot = np.zeros(10)
    one_hot[d] = 1
    res.append(one_hot)
  return np.array(res)

@np.vectorize
def toScalar(data):
  if isinstance(data, Scalar):
    return Scalar(data.data, require_grad=True)
  return Scalar(data, require_grad=True)

@np.vectorize
def toConstant(data):
  if isinstance(data, Scalar):
    return Scalar(data.data, require_grad=False)
  return Scalar(data, require_grad=False)