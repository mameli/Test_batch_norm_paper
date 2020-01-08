from fastai import *
from fastai.vision import *
from fastai.torch_core import add_metrics
from fastai.callback import Callback
from batch_norm_vgg import VGG, make_layers, make_layers_BN


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


class AccuracyValidationList(Callback):
    def __init__(self):
        super().__init__()
        self.name = "Val Acc"
        self.val_accs = []

    def on_epoch_begin(self, **kwargs):
        self.accs = []
        self.currentAcc = 0.

    def on_batch_end(self, last_output, last_target, **kwargs):
        n = last_target.shape[0]
        outs = last_output.argmax(dim=-1).view(n, -1)
        targs = last_target.view(n, -1)
        self.accs.append((outs == targs).float().mean())

    def on_epoch_end(self, last_metrics, **kwargs):
        temp_val = sum(self.accs) / len(self.accs)
        self.val_accs.append(temp_val)
        return add_metrics(last_metrics, temp_val)


@dataclass
class LossVariation(LearnerCallback):
    def __init__(self, learn: Learner):
        super().__init__(learn)
        self.loss_var = []
        self.losses = []
        self.old_loss = 0

    def on_backward_begin(self, last_loss, **kwargs):
        current_loss = last_loss.data.clone()
        self.loss_var.append(
            (self.old_loss - current_loss).norm(2).cpu().numpy())
        self.losses.append(current_loss.cpu().numpy())
        self.old_loss = current_loss


@dataclass
class GradVariation(LearnerCallback):
    def __init__(self, learn: Learner):
        super().__init__(learn)
        self.grad_var = []
        self.grads_l2 = []
        self.old_grad = None
        self.first_step = True

        conv_layer_pos = 5
        conv_counter = 0
        self.grad_layer_pos = 0
        for i, p in enumerate(self.learn.model.features):
            if isinstance(p, nn.modules.Conv2d):
                conv_counter += 1
            if conv_counter == conv_layer_pos:
                self.grad_layer_pos = i
                break

    def on_backward_end(self, **kwargs):
        wg = self.learn.model.features[self.grad_layer_pos].weight.grad
        if not self.first_step:
            self.grad_var.append((self.old_grad + wg).abs().mean())
            self.grads_l2.append((self.old_grad + wg).norm(2))
        self.old_grad = wg.data.clone()
        self.first_step = False
