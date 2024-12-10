import torch
import torch.nn as nn
from mltu.torch.losses import CTCLoss
from mltu.torch.metrics import CERMetric , WERMetric
# CTLoss from mltu adapted to work with the 
# HTR best practices model  
class CTCLossShortcut(CTCLoss): 
    # adapted to work with the HTR best practices model 
    # that has a shortcut connection 
    def forward(self,output,target): 
        if len(output) == 2: 
            return super().forward(output[0],target) + 0.1 * super().forward(output[1],target)
        else : 
            return super().forward(output,target)
        
class CERMetricShortCut(CERMetric): 
    
    def update(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> None:
        if len(output) == 2: 
            output = output[0]
        super().update(output, target, **kwargs)


class WERMetricShortCut(WERMetric):
    def update(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> None:
        if len(output) == 2: 
            output = output[0]
        super().update(output, target, **kwargs)