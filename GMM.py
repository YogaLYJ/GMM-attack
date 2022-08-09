import torch.optim as optim
from GMM_model import *
from common import *
import cv2
import torch.nn.functional as F
import numpy as np

def save_res(pert,pert_img,iter_num):

    def reverse_norm(imgs,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
        imgs=np.asarray(imgs)
        mean=np.array(mean).reshape(3,1,1)
        std=np.array(std).reshape(3,1,1)
        imgs=imgs*std+mean

        imgs=np.clip(imgs,0,1)
        return imgs

    res_dir='results/temp'

    # filename=args.img_path.split('/')[-1]
    saved_file='{}/{}'.format(res_dir,str(iter_num)+'.png')

    pert=pert.data.cpu().numpy()[0]
    pert_img=pert_img.data.cpu().numpy()[0]

    pert_img=reverse_norm(pert_img)

    pert=pert*255/12

    pert=np.tile(pert,(3,1,1))

    img=np.concatenate([pert_img,pert],axis=2)
    cv2.imwrite(saved_file,(img.transpose(1,2,0)*255).astype('uint8'))


# Spatial Information / Image Complexity
def SI(img_path):
    gray_img = cv2.imread(img_path,0)
    JND_mask = JND(gray_img)
    x = cv2.Sobel(gray_img,cv2.CV_16S,1,0)
    y = cv2.Sobel(gray_img,cv2.CV_16S,0,1)
    absX = cv2.convertScaleAbs(x)  # turn to uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    return JND_mask, dst.mean()


# LA in paper
def LA(img):
    m, n = img.shape
    LA_mask = np.zeros((m, n))
    save_list = np.zeros(256)
    for i in range(128):
        save_list[i] = 17 * (1 - np.sqrt(i / 127)) + 3
    for i in range(128,256):
        save_list[i] = 3 * (i - 127) / 128 + 3

    for i in range(m):
        for j in range(n):
            LA_mask[i][j] = save_list[img[i][j]]

    return LA_mask


# CM in paper
def CM(img):
    structural_img = cv2.Canny(img, 50, 150)
    max_img = F.max_pool2d(input=torch.Tensor(structural_img).unsqueeze(0), kernel_size=5, padding=2, stride=1)
    min_img = -F.max_pool2d(input=-torch.Tensor(structural_img).unsqueeze(0), kernel_size=5, padding=2, stride=1)
    C = (max_img - min_img).squeeze(0)
    EM_mask = C * 0.117

    textural_img = img - structural_img
    max_img = F.max_pool2d(input=torch.Tensor(textural_img).unsqueeze(0), kernel_size=5, padding=2, stride=1)
    min_img = -F.max_pool2d(input=-torch.Tensor(textural_img).unsqueeze(0), kernel_size=5, padding=2, stride=1)
    C = (max_img - min_img).squeeze(0)
    TM_mask = C * 0.117 * 3

    CM_mask = EM_mask + TM_mask

    return CM_mask


# Compute JND mask
def JND(img):
    LA_img = torch.Tensor(LA(img))
    CM_img = CM(img)
    JND_mask = LA_img + CM_img - 0.3*torch.min(LA_img,CM_img)

    return JND_mask


# The untargeted adversarial loss
def CW_loss(x,y):
    # get target logits
    target_logits = torch.gather(x, 1, y.view(-1, 1))

    # get largest non-target logits
    max_2_logits, argmax_2_logits = torch.topk(x, 2, dim=1)
    top_max, second_max = max_2_logits.chunk(2, dim=1)
    top_argmax, _ = argmax_2_logits.chunk(2, dim=1)
    targets_eq_max = top_argmax.squeeze().eq(y).float().view(-1, 1)
    targets_ne_max = top_argmax.squeeze().ne(y).float().view(-1, 1)
    max_other = targets_eq_max * second_max + targets_ne_max * top_max
    # Untargeted attack
    f6 = torch.clamp(target_logits - max_other, min=-1*0.1)

    return f6.squeeze()


# parameter clipping
def param_clip(model,config):
    for name,param in model.named_parameters():
        if name in list(config.keys()):
            minv,maxv=config[name]
            param.data.clamp_(minv,maxv)


def GMM_Iter(args, target_model, QA_model, image, image_path, label=None, max_iter=501, pert_scale=12):
    '''
        input:
            target_model: target model to attack
            image: input image
            label: true label
            max_iter: maximum iterations
            pert_scale: scale of perturbations

        output:
            is_adv: inicator for adversary
            adv_cls: predicted class of the pertubated image
            pert: harmonic perturbations
            pert_img: perturbated image
    '''

    # target model setting
    target_model = target_model.cuda()
    target_model = target_model.eval()

    # image,label
    image=image.cuda()
    label=label.cuda()

    # Complexity
    com_map, complexity = SI(image_path)
    I = (np.array(com_map)).flatten().argsort()[::-1]
    max_index = I[0:args.Gaussian_number] # positions with top K JND values

    com_map = (com_map / 500).unsqueeze(0).unsqueeze(0)
    com_map = com_map.cuda()

    # Initialize mu_x, mu_y
    _, _, h, w = image.size()
    P = (max_index + 1) // w
    Q = (max_index + 1) % w
    for i in range(args.Gaussian_number): # positions with top K JND values
        if Q[i] == 0:
            Q[i] = w
        else:
            P[i] = P[i] +1

    x_mu = -1 + 2 * (P - 1) / (h - 1)
    y_mu = -1 + 2 * (Q - 1) / (w - 1)

    # Initialize p
    p = 1/(np.ones(args.Gaussian_number)*args.Gaussian_number)

    # threshold for perceptual loss
    delta_2 = 0.1 * h * w

    target_model.zero_grad()
    x, y = Variable(image, requires_grad=True), Variable(label)

    # define harmonic model
    gmm_model=GMM_Network(target_model,1,x_mu,y_mu,p,pert_scale,args.Gaussian_number)
    gmm_model.cuda()

    # freeze parameters in target_model
    for name, param in gmm_model.named_parameters():
        if param.requires_grad and 'target_model' in name:
            param.requires_grad=False

    # sperate lr for P_a and P_h
    P_a=[]
    P_h=[]
    P_x=[]
    P_y=[]
    for name, param in gmm_model.named_parameters():
        if param.requires_grad:
            if 'p' in name:
                P_h.append(param)
            elif 'mu_x' in name:
                P_x.append(param)
            elif 'mu_y' in name:
                P_y.append(param)
            else:
                P_a.append(param)

    optimizer=optim.Adam([  {'params': P_x},{'params': P_y},{'params': P_a},
                            {'params': P_h, 'lr': 5*args.lr}],
                            lr = args.lr)

    for it in range(max_iter):
        pred, pert, wo_norm, pert_img = gmm_model(x)

        bad_pixels = torch.sum(torch.abs(pert)>com_map)

        dis = QA_model.forward(x,wo_norm).squeeze()

        # Untargeted Attack
        if bad_pixels < delta_2:
            loss = CW_loss(pred, y)
        else:
            loss = 0.5 * CW_loss(pred, y) + args.perceptual_weight * dis

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        param_clip(gmm_model, config.param_range)

        _,cls=pred.data.max(1)
        indi=cls.ne(y.data)

        if indi[0] and dis < 0.2:
            return (True, cls, pert, pert_img, P_x, P_y)

    if indi[0]:
        return (True, cls, pert, pert_img, P_x, P_y)
    else:
        return (False, cls, pert, pert_img, P_x, P_y)
