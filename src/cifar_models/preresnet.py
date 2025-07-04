import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..noisy_mixup import do_noisy_mixup

class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out
    

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class VIT(nn.Module):
    def __init__(self, vit: nn.Module, preprocessor, num_classes=10):
        super().__init__()
        self.vit = vit
        self.preprocessor = preprocessor
        self.classifier = nn.Linear(197 * 768, num_classes)


    def forward(self, x, targets=None, jsd=0, mixup_alpha=0.0, manifold_mixup=0, 
                add_noise_level=0.0, mult_noise_level=0.0, sparse_level=1.0, numpy_gen=None, torch_gen=None):
        assert(not manifold_mixup)
        if mixup_alpha > 0.0:
            k = 0
            if torch_gen is None:
                torch_gen = torch.Generator(device='cuda')
            if numpy_gen is None:
                numpy_gen = np.random.default_rng()
        else:
            k = -1

        if mixup_alpha > 0.0 and manifold_mixup == True: 
            k = numpy_gen.choice(range(len(self.blocks)), 1)[0]
        if k == 0: # Do input mixup if k is 0 
          # Clone is necessary for KD because otherwise the original images are modified in place!
          x, targets_a, targets_b, lam = do_noisy_mixup(x.clone(), targets, numpy_gen, torch_gen, jsd=jsd, alpha=mixup_alpha, 
                                              add_noise_level=add_noise_level, 
                                              mult_noise_level=mult_noise_level,
                                              sparse_level=sparse_level)
        
        x = ((x + 1.0) / 2).clamp(0.0, 1.0)
        # print(x)
        inputs = self.preprocessor(images=x, return_tensors="pt", do_normalize=False, do_rescale=False).to(x.device)
        outputs = self.vit(**inputs)
        out = outputs.last_hidden_state
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        
        if mixup_alpha > 0.0:
            return out, targets_a, targets_b, lam
        else:
            return out



class ResNetBase(nn.Module):
    def __init__(self):
        super().__init__()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, targets=None, jsd=0, mixup_alpha=0.0, manifold_mixup=0, 
                add_noise_level=0.0, mult_noise_level=0.0, sparse_level=1.0, numpy_gen=None, torch_gen=None):
        
        if mixup_alpha > 0.0:
            k = 0
            if torch_gen is None:
                torch_gen = torch.Generator(device='cuda')
            if numpy_gen is None:
                numpy_gen = np.random.default_rng()
        else:
            k = -1


        if mixup_alpha > 0.0 and manifold_mixup == True: 
            k = numpy_gen.choice(range(len(self.blocks)), 1)[0]
        if k == 0: # Do input mixup if k is 0 
          # Clone is necessary for KD because otherwise the original images are modified in place!
          x, targets_a, targets_b, lam = do_noisy_mixup(x.clone(), targets, numpy_gen, torch_gen, jsd=jsd, alpha=mixup_alpha, 
                                              add_noise_level=add_noise_level, 
                                              mult_noise_level=mult_noise_level,
                                              sparse_level=sparse_level)

        out = self.conv1(x)
        
        for i, ResidualBlock in enumerate(self.blocks):
            out = ResidualBlock(out)
            if k == (i+1): # Do manifold mixup if k is greater 0
                out, targets_a, targets_b, lam = do_noisy_mixup(out, targets, numpy_gen, torch_gen, jsd=jsd, alpha=mixup_alpha, 
                                           add_noise_level=add_noise_level, 
                                           mult_noise_level=mult_noise_level,
                                           sparse_level=sparse_level)
                
        if hasattr(self, "bn1"): # For WideResNet28
            out = F.relu(self.bn1(out))
                
        out = F.avg_pool2d(out, out.size(dim=3)) # 4 for the normal models
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        if mixup_alpha > 0.0:
            return out, targets_a, targets_b, lam
        else:
            return out

# class Ensemble(nn.Module):
#     def __init__(self, models: list[nn.Module]):
#         super().__init__()
#         self.models = nn.ModuleList(models)

#     def forward(self, x, **kwargs):
#         soft = nn.Softmax(dim=1)
#         data = []
#         probs = []
#         for model in self.models:
#             output = model(x, **kwargs)
#             if isinstance(output, tuple):
#                 data.append(output[0])
#             else:
#                 data.append(output)
            
#             probs.append(soft(data[-1]))
            
#         offset = data[0][0] - torch.log(probs[0][0])
#         probs = torch.stack(probs, dim=0)
#         average = torch.mean(probs, dim=0)
#         average = torch.log(average) + offset
#         # Average Probs 
#         return average

class Ensemble(nn.Module):
    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, **kwargs):
        data = []
        for model in self.models:
            output = model(x, **kwargs)
            if isinstance(output, tuple):
                data.append(output[0])
            else:
                data.append(output)
        data = torch.stack(data, dim=0)
        average = torch.mean(data, dim=0)
        # Average logits 
        return average
        

class PreActResNet(ResNetBase):
    def __init__(self, block, num_blocks, num_classes=10, width=1):
        super(PreActResNet, self).__init__()
        
        widths = [int(w * width) for w in [64, 128, 256, 512]]
        
        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(widths[3]*block.expansion, num_classes)
        self.blocks = [self.layer1, self.layer2, self.layer3, self.layer4]

class SmallPreActResNet(ResNetBase):
    def __init__(self, block, num_blocks, num_classes=10, width=1):
        super().__init__()
        
        widths = [int(w * width) for w in [16, 32, 64]]
        
        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.linear = nn.Linear(widths[-1]*block.expansion, num_classes)
        self.blocks = [self.layer1, self.layer2, self.layer3]

def PreActResNet20(**kwargs):
    return SmallPreActResNet(PreActBlock, [3, 3, 3], **kwargs)

def PreActResNet32(**kwargs):
    return SmallPreActResNet(PreActBlock, [5, 5, 5], **kwargs)

def PreActResNet18(**kwargs):
    return PreActResNet(PreActBlock, [2,2,2,2], **kwargs)

def PreActWideResNet18(**kwargs):
    return PreActResNet(PreActBlock, [2,2,2,2], width=2, **kwargs)

def PreActResNet34(**kwargs):
    return ResNet(BasicBlock, [3,4,6,3], **kwargs)

preactresnet18 = PreActResNet18
preactwideresnet18 = PreActWideResNet18
preactresnet34 = PreActResNet34
preactresnet20 = PreActResNet20
preactresnet32 = PreActResNet32

def test():
    net = PreActResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())