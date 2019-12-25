from fastai import *
from fastai.vision import *
from batch_norm_vgg import VGG, make_layers


@dataclass
class ICS(LearnerCallback):
    def __init__(self, learn: Learner, num_classes, bn):
        super().__init__(learn)
        self.cos = nn.CosineSimilarity(dim=0)
        self.criterion = nn.CrossEntropyLoss()
        self.bn = bn
        self.num_classes = num_classes
        self.dummyNet = VGG(make_layers(batch_norm=self.bn),
                            num_classes=self.num_classes).cuda()

    def on_train_begin(self, **kwargs):
        self.cos_values = []
        self.ics_values = []

    def on_batch_begin(self, **kwargs):
        for i, p in enumerate(self.dummyNet.features):
            if isinstance(p, nn.modules.Conv2d):
                p.weight.data = self.learn.model.features[i].weight.data.clone(
                )
        self.dummyNet.zero_grad()

    def on_step_end(self, last_input, last_target, **kwargs):
        lr = self.learn.opt.lr
        conv_layer_pos = 5
        conv_counter = 0
        ics_layer_pos = 0
        for i, p in enumerate(self.dummyNet.features):
            if isinstance(p, nn.modules.Conv2d):  # params before batchNorm
                p.weight.data.sub_(
                    self.learn.model.features[i].weight.grad * lr)
                conv_counter += 1
            if conv_counter == conv_layer_pos:
                ics_layer_pos = i
                break

        dummy_Outputs = self.dummyNet(last_input)
        dummy_Loss = self.criterion(dummy_Outputs, last_target)
        dummy_Loss.backward()
        ics = (self.learn.model.features[ics_layer_pos].weight.grad -
               self.dummyNet.features[ics_layer_pos].weight.grad).norm(2)
        self.ics_values.append(ics.cpu())

        w = torch.flatten(self.learn.model.features[ics_layer_pos].weight.grad)
        dw = torch.flatten(self.dummyNet.features[ics_layer_pos].weight.grad)
        self.cos_values.append(self.cos(dw, w).cpu())


@dataclass
class AccuracyList(LearnerCallback):
    def __init__(self, learn: Learner):
        super().__init__(learn)

    def on_train_begin(self, **kwargs):
        self.accs = []

    def on_step_end(self, last_output, last_target, **kwargs):
        n = last_target.shape[0]
        outs = last_output.argmax(dim=-1).view(n, -1)
        targs = last_target.view(n, -1)
        self.accs.append((outs == targs).float().mean())
