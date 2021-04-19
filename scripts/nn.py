import torch
import torchvision


def extract_features(model):
    """Modifies the model to return (prediction, features) rather than just
    prediction.
    The change occurs in-place, but the new model is also returned."""

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    if (isinstance(model, torchvision.models.AlexNet) or
        isinstance(model, torchvision.models.VGG)):
        model.classifier[-1] = InputExtractor(model.classifier[-1])
    elif (isinstance(model, torchvision.models.ResNet) or
          isinstance(model, torchvision.models.Inception3) or
          isinstance(model, torchvision.models.video.resnet.VideoResNet)):
        model.fc = InputExtractor(model.fc)
    elif (isinstance(model, torchvision.models.DenseNet) or
          isinstance(model, torchvision.models.MobileNetV2)):
        model.classifier = InputExtractor(model.classifier)
    else:
        raise NotImplementedError()

    return model

_extracting = True

class InputExtractor(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        if _extracting:
            return self.module(x), x
        return self.module(x)

class set_extract_enabled(object):
    # Based on set_grad_enabled https://pytorch.org/docs/stable/_modules/torch/autograd/grad_mode.html#no_grad
    def __init__(self, mode: bool) -> None:
        global _extracting
        self.prev = _extracting
        _extracting = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        global _extracting
        _extracting = self.prev
