from engine.autograd import Scalar
from engine.utils import init_like, elementwise_lst_op , elementwise_op

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

class Adam:
  def __init__(self, parameters, alpha=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8):
    self._parameters = parameters
    self._alpha = alpha
    self._beta_1 = beta_1
    self._beta_2 = beta_2
    self._eps = eps 

    self._m = init_like(self._parameters, 0.0)
    self._v = init_like(self._parameters, 0.0)

    self._t = 0

  def step(self):
    g = elementwise_op(self._parameters, lambda x: x.grad)
    self._t += 1

    self._m = elementwise_lst_op(
      elementwise_op(self._m, lambda x: x*self._beta_1),
      elementwise_op(g, lambda x: x*(1-self._beta_1)),
      lambda a,b: a+b)
    
    self._v = elementwise_lst_op(
      elementwise_op(self._v, lambda x: x*self._beta_2),
      elementwise_op(g, lambda x: (x**2)*(1-self._beta_2)),
      lambda a,b: a+b)
    
    m_hat = elementwise_op(self._m, lambda a: a/(1-self._beta_1**self._t))
    v_hat = elementwise_op(self._v, lambda a: a/(1-self._beta_2**self._t))

    updates = elementwise_lst_op(m_hat, v_hat, lambda m,v: self._alpha*m/(self._eps + v**0.5))

    def update_fn(p, u):
      p.data = p.data - u

    elementwise_lst_op(self._parameters, updates, update_fn)


                                 

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
