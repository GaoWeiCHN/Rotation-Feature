from torch.nn import Module
from gcn.functions.rotationF import ROT_Function

class ROT_Module(Module):
    def forward(self, feature):
        return ROT_Function.apply(feature)