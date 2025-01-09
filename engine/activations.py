import numpy as np
from engine.autograd import Scalar
from engine.ops import exp

@np.vectorize
def sigmoid(z):
  # print(z[0])
  out = Scalar(1 / (1 + np.exp(-z.data)), children=[z])

  def _backward():
    z.grad += out.grad * out.data * (1 - out.data)

  out._backward = _backward
  return out

@np.vectorize
def leakyRelu(in_param, alpha = 0.01):
  out = Scalar(max(in_param.data, in_param.data), [in_param])

  def _backward():
    in_param.grad += out.grad if in_param.data > 0 else out.data*alpha

  out._backward = _backward
  return out


def softmax(in_params):
  result = []
  for in_param in in_params:
    out = []
    # print(in_param.shape)
    in_data = [v.data for v in in_param]
    m = max(in_data)
    in_data = [v - m for v in in_data]
    # print(in_param.shape)
    exp_data = exp(in_param - Scalar(m))
    exp_sum = np.sum(exp_data)

    for exp_i in exp_data:
      # val = Scalar(exp_i / exp_sum, in_param)

      # def _backward():
      #   param_i.grad += val.grad * val.data * (1 - val.data)

      # val._backward = _backward
      # out =
      out.append(exp_i / exp_sum)
    # print(exp_data)
    result.append(out)
  return np.array(result)