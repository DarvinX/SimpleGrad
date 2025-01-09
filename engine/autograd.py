class Scalar:
  def __init__(self, data, children=[], require_grad = True):
    self.data = data
    self.require_grad = require_grad
    self.grad = 0.0
    self._backward = lambda: None
    self.children = set(children)
    self.new_data = 0.0

  def __repr__(self):
    return f"({self.data}, grad: {self.grad}, req_grad: {self.require_grad})"

  def __add__(self, other):
    out = Scalar(self.data + other.data, children=[self, other])

    def _backward():
      # print("called (+)", self, other, out)
      self.grad += out.grad
      other.grad += out.grad

    out._backward = _backward
    return out

  def __sub__(self, other):
    out = Scalar(self.data - other.data, children=[self, other])

    def _backward():
      # print("called (-)", self, other, out)
      self.grad += out.grad
      other.grad -= out.grad

    out._backward = _backward
    return out

  def __truediv__(self, other):
    out = Scalar(self.data/other.data, children=[self,other])

    def _backward():
      self.grad += out.grad/other.data
      other.grad += self.data*(-1/other.data**2)*out.grad

    out._backward = _backward
    return out

  def __mul__(self, other):
    out = Scalar(self.data * other.data, children=[self, other])

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad

    out._backward = _backward
    return out

  def backward(self):
    # Toposort
    res = []
    self.grad = 1.0

    def topo(root):
      #print(root)
      if root in res or root.require_grad == False:
        return
      for ch in root.children:
        # if ch.require_grad:
        topo(ch)
      res.append(root)
    topo(self)
    for n in reversed(res):
      n._backward()
    # print(res)
