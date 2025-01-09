from engine.autograd import Scalar

class SGD:
  def __init__(self, parameters, lr):
    self._parameters = parameters
    self._lr = lr

  def step(self):
    # print(self._parameters)
    def update(params):
      if isinstance(params, Scalar):
        params.data -= params.grad*self._lr
        # print("update", params.data)
      else:
        for param in params:
          update(param)

    update(self._parameters)

  def zero_grad(self):
    def zero(params):
      # print(type(params))
      if isinstance(params, Scalar):
        params.grad = 0.0
        # print("update", param.data)
      else:
        for param in params:
          zero(param)

    zero(self._parameters)