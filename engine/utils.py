import numpy as np
from engine.autograd import Scalar
from collections.abc import Iterable

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


def elementwise_lst_op(lst_1, lst_2, op):
  if isinstance(lst_1, Iterable) and isinstance(lst_2, Iterable):
    return [elementwise_lst_op(l,v,op) for l,v in zip(lst_1,lst_2)]
  elif not isinstance(lst_1, Iterable) and not isinstance(lst_2, Iterable):
    return op(lst_1, lst_2)
  else:
    ValueError("shapes do not match.")

def elementwise_op(lst_1, op):
  if isinstance(lst_1, Iterable):
    return [elementwise_op(l,op) for l in lst_1]
  else:
    return op(lst_1)

def init_like(lst, val):
  return elementwise_op(lst, lambda a: val)