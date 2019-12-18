from fastai import *
from fastai.vision import *
from batch_norm_vgg import VGG, make_layers

@dataclass
class ICS(LearnerCallback):
    def __init__(self, learn:Learner, num_classes, bn):
        super().__init__(learn)
        self.cos = nn.CosineSimilarity(dim=0)
        self.criterion = nn.CrossEntropyLoss()
        self.bn = bn
        self.num_classes = num_classes
    
    def on_train_begin(self, **kwargs):
        self.cos_values = []
        self.ics_values = []
        
    def on_batch_begin(self, **kwargs):
        self.conv2Weight = self.learn.model.features[8].weight.data.clone()
        self.dummyNet = VGG(make_layers(batch_norm=self.bn), num_classes=self.num_classes).cuda()
        netParams = list(self.learn.model.parameters())
        for idx, p in enumerate(self.dummyNet.parameters()):
            p.data = netParams[idx].data.clone()
        
    def on_step_end(self, last_input, last_target, **kwargs):
        lr = self.learn.opt.lr
        for i, p in enumerate(self.dummyNet.features):
            if isinstance(p, nn.modules.Conv2d): # params before batchNorm
                p.weight.data.sub_(self.learn.model.features[i].weight.grad * lr)
            if isinstance(p, nn.modules.BatchNorm2d):
                break
            
        dummyOutputs = self.dummyNet(last_input) 
        dummyLoss = self.criterion(dummyOutputs, last_target)
        dummyLoss.backward()
        ics = (self.learn.model.features[8].weight.grad - self.dummyNet.features[8].weight.grad).norm(2)
        self.ics_values.append(ics)
        
        w  = torch.flatten(self.learn.model.features[8].weight.grad)
        dw = torch.flatten(self.dummyNet.features[8].weight.grad)
        self.cos_values.append(self.cos(w, dw)) 
        
@dataclass
class AccuracyList(LearnerCallback):
    def __init__(self, learn:Learner):
        super().__init__(learn)
    
    def on_train_begin(self, **kwargs):
        self.accs = []
        
    def on_step_end(self, last_output, last_target, **kwargs):
        n = last_target.shape[0]
        outs = last_output.argmax(dim=-1).view(n,-1)
        targs  = last_target.view(n,-1)
        self.accs.append((outs==targs).float().mean())