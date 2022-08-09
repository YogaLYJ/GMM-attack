
import torchvision

def load_model(name,pretrained=True):
    if name=='resnet50':
        model=torchvision.models.resnet50(pretrained=pretrained)
    elif name=='vgg16':
        model=torchvision.models.vgg16(pretrained=pretrained)
    elif name=='inceptionv3':
        model=torchvision.models.inception_v3(pretrained)
    elif name=='densenet121':
        model=torchvision.models.densenet121(pretrained)
    else:
        raise NotImplementedError('{} is not allowed!'.format(name))

    return model
