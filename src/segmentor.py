import torch
import torch.nn as nn

from src.loss import CrossEntropyLoss
from src.lovasz import LovaszLoss
from src.model import PointTransformerV3, Point

LOSS_FUNCTIONS = {"CrossEntropyLoss": CrossEntropyLoss,
                  "LovaszLoss": LovaszLoss}

class Criteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            loss_type = loss_cfg.pop("type")
            if loss_type in LOSS_FUNCTIONS:
                self.criteria.append(LOSS_FUNCTIONS[loss_type](**loss_cfg))

    def __call__(self, pred, target):
        if len(self.criteria) == 0:
            # loss computation occur in model
            return pred
        loss = 0
        for c in self.criteria:
            loss += c(pred, target)
        return loss
    

class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        
        if backbone != None:
            # model config
            backbone.pop('type')
            # initialize model with loaded config
            self.backbone = PointTransformerV3(**backbone)
        else:
            self.backbone = nn.Identity()

        self.criteria = Criteria(criteria)

        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        seg_logits = self.seg_head(point.feat)
        
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point

        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits

        return return_dict
