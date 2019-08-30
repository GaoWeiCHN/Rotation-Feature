from torch.autograd import Function
from torch.autograd import Variable
from .._ext import mr

class ROT_Function(Function):
    @staticmethod
    def forward(ctx, feature):
        output = feature.new()
        mr.feature_rotation(feature, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.new()
        mr.grad_rotation(grad_output, output)
        return Variable(output)