from engine.autograd import Scalar
import numpy as np

def log(val):
  # print(val)
  e = 1e-9
  out = Scalar(np.log(val.data+e), children=[val])
  # print(out

  def _backward():
    # print("im called", out.grad)
    val.grad += out.grad / (val.data+e)

  out._backward = _backward
  return out

@np.vectorize
def exp(val):
  out = Scalar(np.exp(val.data), children=[val])

  def _backward():
    val.grad += out.grad * out.data

  out._backward = _backward
  return out