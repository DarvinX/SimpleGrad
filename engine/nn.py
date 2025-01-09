from engine.utils import toScalar
import numpy as np
import matplotlib.pyplot as plt

class Linear:
  def __init__(self, in_dim, out_dim, activation=lambda x: x):
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
      return f"Linear(in_dim: {self.in_dim}, out_dim: {self.out_dim}, activation: {self.activation.__name__})"
  
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