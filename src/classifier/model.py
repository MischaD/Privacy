from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchxrayvision.models import model_urls, _Transition, _DenseBlock, op_norm, get_weights
import torchxrayvision as xrv


class DenseNet(nn.Module):
    """Based on 
    `"Densely Connected Convolutional Networks" <https://arxiv.org/abs/1608.06993>`_
    and
    <https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/models.py>

    Possible weights for this class include:

    .. code-block:: python

        ## 224x224 models
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        model = xrv.models.DenseNet(weights="densenet121-res224-rsna") # RSNA Pneumonia Challenge
        model = xrv.models.DenseNet(weights="densenet121-res224-nih") # NIH chest X-ray8
        model = xrv.models.DenseNet(weights="densenet121-res224-pc") # PadChest (University of Alicante)
        model = xrv.models.DenseNet(weights="densenet121-res224-chex") # CheXpert (Stanford)
        model = xrv.models.DenseNet(weights="densenet121-res224-mimic_nb") # MIMIC-CXR (MIT)
        model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch") # MIMIC-CXR (MIT)

    :param weights: Specify a weight name to load pre-trained weights
    :param op_threshs: Specify a weight name to load pre-trained weights 
    :param apply_sigmoid: Apply a sigmoid 

    """
    targets: List[str] = [
        'Atelectasis',
        'Consolidation',
        'Infiltration',
        'Pneumothorax',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Effusion',
        'Pneumonia',
        'Pleural_Thickening',
        'Cardiomegaly',
        'Nodule',
        'Mass',
        'Hernia',
        'Lung Lesion',
        'Fracture',
        'Lung Opacity',
        'Enlarged Cardiomediastinum',
    ]
    """"""

    def __init__(self,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=18,
                 in_channels=1,
                 weights=None,
                 op_threshs=None,
                 apply_sigmoid=False
                 ):

        super(DenseNet, self).__init__()

        self.apply_sigmoid = apply_sigmoid
        self.weights = weights
        
        self.transforms = torchvision.transforms.Compose([
            lambda x: (x * 1024), 
            #torchvision.transforms.Resize(224, torchvision.transforms.InterpolationMode.BILINEAR),
            #torchvision.transforms.CenterCrop(224),
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(512),
            lambda x: x.clip(-1024, 1024),
            torchvision.transforms.ToTensor(),
        ])

        if self.weights is not None:
            if not self.weights in model_urls.keys():
                possible_weights = [k for k in model_urls.keys() if k.startswith("densenet")]
                raise Exception("Weights value must be in {}".format(possible_weights))

            # set to be what this model is trained to predict
            self.targets = model_urls[weights]["labels"]
            self.pathologies = self.targets  # keep to be backward compatible

            # if different from default number of classes
            if num_classes != 18:
                raise ValueError("num_classes and weights cannot both be specified. The weights loaded will define the own number of output classes.")

            num_classes = len(self.pathologies)

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        # needs to be register_buffer here so it will go to cuda/cpu easily
        self.register_buffer('op_threshs', op_threshs)

        if self.weights != None:
            self.weights_filename_local = get_weights(weights)

            try:
                savedmodel = torch.load(self.weights_filename_local, map_location='cpu')
                # patch to load old models https://github.com/pytorch/pytorch/issues/42242
                for mod in savedmodel.modules():
                    if not hasattr(mod, "_non_persistent_buffers_set"):
                        mod._non_persistent_buffers_set = set()

                self.load_state_dict(savedmodel.state_dict())
            except Exception as e:
                print("Loading failure. Check weights file:", self.weights_filename_local)
                raise e

            self.eval()

            if "op_threshs" in model_urls[weights]:
                self.op_threshs = torch.tensor(model_urls[weights]["op_threshs"])

            self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def __repr__(self):
        if self.weights is not None:
            return "XRV-DenseNet121-{}".format(self.weights)
        else:
            return "XRV-DenseNet"

    def features2(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        return out

    def forward(self, x):
        # assumes -1 to 1, torch tensor of shape B x 3 x H x W 
        x = x.mean(axis=1, keepdim=False)
        x = x.numpy()
        x = self.transforms(x).transpose(0,1).transpose(1,2).unsqueeze(dim=0)
        print(x.size())
        print(x.min())
        print(x.max())

        # feature extraction 
        features = self.features2(x)

        # classification 
        out = self.classifier(features)

        if hasattr(self, 'apply_sigmoid') and self.apply_sigmoid:
            out = torch.sigmoid(out)

        if hasattr(self, "op_threshs") and (self.op_threshs != None):
            out = torch.sigmoid(out)
            out = op_norm(out, self.op_threshs)
        return out
