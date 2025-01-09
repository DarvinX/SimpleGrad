import numpy as np
from engine.autograd import Scalar
from engine.ops import log



def mse(y_true, y_pred):
  #print(y_pred, y_true)
  # y_pred.squeeze()
  # y_true.squeeze()
  # error = np.array([p.data - t.data for p,t in zip(y_pred, y_true)])
  error = y_pred - y_true
  # print("error",error)
  mse = np.dot(error.T, error)*Scalar(1/len(error))
  # def _backward():
  #   # print("mse backprop")
  #   for p,t in zip(y_pred, y_true):
  #     p.grad += 2 * (p.data - t.data) / len(error)

  # mse._backward = _backward
  return mse #[0][0]


def cross_entropy(y_true, y_pred):
  res = []
  for b in range(y_pred.shape[0]):
    loss = np.sum([y*log(p) for y,p in zip(y_true[b], y_pred[b])])*Scalar(-1)
    res.append(loss)
  return np.sum(res)/Scalar(len(res), require_grad=False)