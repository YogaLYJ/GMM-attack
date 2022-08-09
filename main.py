import cv2
import torch
import torchvision.transforms as T
import torchvision.models as tmodels
import argparse

from GMM import GMM_Iter
from threatmodels import load_model
from util_save import save_res
import models


def pre_process(img):
    trans = T.Compose([T.ToPILImage(),T.Resize(256),T.CenterCrop(224),T.ToTensor()])

    return trans(img)


# demo for non-targeted attack
def main(args):
    torch.backends.cudnn.deterministic = True

    # QA model
    QA_model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=True)

    # source model
    model = load_model(args.model)

    # Target model
    target_model = tmodels.vgg16(pretrained=True)
    target_model.eval()
    target_model = target_model.cuda()

    # Get image and its groundtruth
    img_path = './test_images/{}'.format(args.image)
    label = args.true_label
    label = torch.ones(1) * int(label)
    label = label.long()

    # Temporarily stored
    img = cv2.imread(img_path)
    img = pre_process(img)
    name = 'temp.png'
    cv2.imwrite(name, (img.data.numpy().transpose(1,2,0) * 255).astype('uint8'))

    img=img.unsqueeze(0)

    is_adv,pred_cls,pert,pert_img,mu_x,mu_y = GMM_Iter(args,model,QA_model,img,name,label=label,pert_scale=args.pert_scale)

    if is_adv:
        save_res(args, pert, pert_img, args.image)
        print("Congratulations!!! An adversarial example is successfully generated!")

        # Test transfer
        pred_trans = target_model(pert_img)
        _, cls = pred_trans.data.max(1)
        if not int(cls.cpu()) == args.true_label:
            print("Congratulations! The adversarial image {} successfully transfers from {} to VGG16!!!".format(args.image, args.model))
        else:
            print("Ooooops! The adversarial image {} fails to transfer from {} to VGG16.".format(args.image[:-5]+'.png', args.model))
    else:
        print("Ooooops! Faile to generate an adversarial example by GMM!")

if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--model',help='source model: vgg16, resnet50, inceptionv3', type=str, default='resnet50')
    parser.add_argument('--image',help='the image to be attacked',type=str,default='ILSVRC2012_val_00000724.JPEG')
    parser.add_argument('--true_label',help='groundtruth of the image',type=int,default=449)
    parser.add_argument('--lr',help='base learning rate', type=float, default=0.01)
    parser.add_argument('--save_dir',help='results saving diresctory', type=str, default='./')
    parser.add_argument('--Gaussian_number', type=int, default=20)
    parser.add_argument('--pert_scale',help='scale of perturbations',type=float, default=12)
    parser.add_argument('--perceptual_weight', type=int, default=100)
    args=parser.parse_args()

    main(args)
