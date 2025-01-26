from engine.utils import toScalar, elementwise_lst_op, elementwise_op, init_like, toConstant
import numpy as np
import matplotlib.pyplot as plt
from engine.autograd import Scalar


class Layer:
  def __init__(self):
    self.trainingMode = True
    self.debugMode = False

  def train(self):
    self.trainingMode = True
  
  def eval(self):
    self.trainingMode = False

  def debugMsg(self, msg):
    if self.debugMode:
      print(msg)

  def debug(self):
    self.debugMode = True

class Linear(Layer):
  def __init__(self, in_dim, out_dim, activation=lambda x: x):
    super().__init__()

    self.in_dim = in_dim
    self.out_dim = out_dim
    self.weights = toScalar(np.random.randn(in_dim, out_dim) * np.sqrt(2 / (in_dim + out_dim)))
    self.bias = toScalar(np.zeros(out_dim))

    # self._parameters = [self.weights, self.bias]
    self.activation = activation

  def __call__(self, x):
    # print("before activation", (np.matmul(x, self.weights) + self.bias).shape)
    return self.activation(np.matmul(x, self.weights) + self.bias)

  def parameters(self):
    return [self.weights, self.bias]

  def __repr__(self):
      return f"\nLinear(in_dim: {self.in_dim}, out_dim: {self.out_dim}, activation: {self.activation.__name__})"
  
class Sequence:
  def __init__(self, layers):
    self.layers = layers

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)

    return x #np.squeeze(x)

  def parameters(self):
    # params = []

      # params.append(layer.parameters())
    return [layer.parameters() for layer in self.layers]
  
  def __getitem__(self, index):
    return self.layers[index]
  
  def __repr__(self):
    return f"Sequence({self.layers})"
  
class LayerNorm(Layer):
  def __init__(self, eps=1e-8):
    super().__init__()
    self.beta = Scalar(0)
    self.gamma = Scalar(1)
    self.eps = Scalar(eps, require_grad=False)

  def __call__(self, x):

    def sumScalar(data):
      r = Scalar(0, require_grad=False)

      for d in data:
        r+=d
      return r

    sample_len = Scalar(len(x[0]), require_grad=False)

    u = [sumScalar(s)/sample_len for s in x]

    var = [
      sumScalar(
        elementwise_op(s, lambda d: (d - u_)**2)
        )/sample_len
        for s,u_ in zip(x,u)]

    x_hat = [elementwise_op(s, lambda d: (d-u_)/((v_ + self.eps)**0.5)) for s, u_, v_ in zip(x,u,var)]

    return elementwise_op(x_hat, lambda d: d*self.gamma + self.beta)
  
  def parameters(self):
    # print([self.gamma, self.beta])
    return [self.gamma, self.beta]
  
  def __repr__(self):
      return f"\nLayerNorm"
  
@np.vectorize
def extract_grad(val):
  return val.grad

@np.vectorize
def extract_value(val):
  return val.data

def plot_gradients(model):
  for layer in model.layers:
    if isinstance(layer, Linear):
      plt.imshow(extract_grad(layer.weights), cmap='hot')
      plt.show()
      print("min: ", min(extract_grad(layer.weights).flatten()))
      print("max: ", max(extract_grad(layer.weights).flatten()))


def plot_weights(model):
  for layer in model.layers:
    if isinstance(layer, Linear):
      plt.imshow(extract_value(layer.weights), cmap='hot')
      plt.show()
      print("min: ", min(extract_grad(layer.weights).flatten()))
      print("max: ", max(extract_grad(layer.weights).flatten()))