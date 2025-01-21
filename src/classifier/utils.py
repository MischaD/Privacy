import torch
from torchvision.models import resnet50
from src.classifier.module import LightningSAFClassifier
from torchvision import transforms
from src.data.inpainter import get_inpainter
from einops import repeat


def get_classifier_forward(config, return_seperate=False):
    """Loads fairnes classifier according to config and path given by config.fairness_classifier_path"""
    pl_model = LightningSAFClassifier.load_from_checkpoint(
        config.af_classifier_path
    ).to("cuda:0")
    pl_model = pl_model.eval()

    pl_model_id = LightningSAFClassifier.load_from_checkpoint(
        config.id_classifier_path
    ).to("cuda:0")
    pl_model_id = pl_model_id.eval()

    def fairness_forward(x):
        if x.min() >= 0:
            x = x * 2 - 1

        if x.size()[-1] != config.data.image_size:
            x = transforms.Resize(config.data.image_size)(x).clip(-1, 1)

        batch = {"image": x.to("cuda")}
        pl_model.eval()
        pl_model.model.eval()
        with torch.no_grad():
            predictions_batch, y_hats_batch = pl_model.predict(batch)

        pl_model_id.eval()
        pl_model_id.model.eval()
        with torch.no_grad():
            predictions_batch_id, y_hats_batch_id = pl_model_id.predict(batch)

        if not return_seperate:
            return torch.logical_and(predictions_batch, predictions_batch_id)
        else:
            return {"id": predictions_batch_id, "af": predictions_batch}

    return fairness_forward
