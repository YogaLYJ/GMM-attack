import os
import cv2
import numpy as np


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


'''
    Save orginal image, perturbated image and perturbation
'''
def save_res(args,pert,pert_img,filename):

    def reverse_norm(imgs,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
        imgs=np.asarray(imgs)
        mean=np.array(mean).reshape(3,1,1)
        std=np.array(std).reshape(3,1,1)
        imgs=imgs*std+mean

        imgs=np.clip(imgs,0,1)
        return imgs

    res_dir_pert_image='{}/Pert_Image_{}'.format(args.save_dir, args.model)
    res_dir_pert = '{}/Pert_{}'.format(args.save_dir, args.model)
    ensure_dir(res_dir_pert_image)
    ensure_dir(res_dir_pert)

    saved_file_image='{}/{}'.format(res_dir_pert_image,filename[:-5]+'.png')
    saved_file_pert = '{}/{}'.format(res_dir_pert, filename[:-5] + '.png')

    # img=np.asarray(org_img)[0]
    pert=pert.data.cpu().numpy()[0]
    pert_img=pert_img.data.cpu().numpy()[0]
    pert_img=reverse_norm(pert_img)

    pert=pert*255/args.pert_scale

    pert=np.tile(pert,(3,1,1))

    # img=np.concatenate([img,pert_img,pert],axis=2)
    cv2.imwrite(saved_file_image,(pert_img.transpose(1,2,0)*255).astype('uint8'))
    cv2.imwrite(saved_file_pert, (pert.transpose(1, 2, 0) * 255).astype('uint8'))
