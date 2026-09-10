
import os
import math
import faiss
import scipy.io
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.log_utils import save_w
from criteria import l2_loss
from reid_model.model import ft_net
from training.coaches.base_coach import BaseCoach
from configs import paths_config, hyperparameters, global_config

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode, Normalize

class LatentRAG(BaseCoach):
    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

#----------------------------------------------------

    def MI_FGSM(
        self,
        generator,
        reid_model,
        lc_modified_coarse,
        lc_fine,
        feat_orig,
        epsilon,
        alpha,
        lambda_id,
        momentum,
        num_iter
    ):
        adv_latent = lc_fine.clone().detach()
        grad_m = torch.zeros_like(adv_latent)

        for i in range(num_iter):
            adv_latent.requires_grad_(True)
            latent_code = torch.cat([lc_modified_coarse, adv_latent], dim=1)
            img_adv  = generator.synthesis(latent_code, noise_mode='const')
            gimg = (resize(img_adv, size=(256, 128), interpolation=InterpolationMode.BICUBIC) + 1) / 2
            imagenet_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            gimg = imagenet_normalize(gimg)
            feat_adv = reid_model(gimg)
            cos_sim = F.cosine_similarity(feat_adv, feat_orig, dim=1)
            loss = 1.0 - cos_sim.mean()
            loss.backward()
            g = adv_latent.grad.data
            norm = g.abs().view(g.shape[0], -1).mean(dim=1, keepdim=True)
            g = g / norm.clamp(min=1e-12).view(-1,1,1)
            grad_m = momentum * grad_m + g
            adv_latent.data -= alpha * grad_m.sign()
            adv_latent.data = adv_latent.data
            adv_latent.grad.zero_()
        return adv_latent.detach()

#----------------------------------------------------

    def MI_FGSM_l2(
        self,
        generator,
        lc_coarse,
        lc_fine,
        real_images_batch,
        epsilon,
        alpha,
        lambda_id,
        momentum,
        num_iter
    ):
        adv_latent = lc_coarse.clone().detach()
        grad_m = torch.zeros_like(adv_latent)

        for i in range(num_iter):
            adv_latent.requires_grad_(True)
            latent_code = torch.cat([adv_latent, lc_fine], dim=1)
            img_adv  = generator.synthesis(latent_code, noise_mode='const')
            l2_loss_val = l2_loss.l2_loss(img_adv, real_images_batch)
            l2_loss_val.backward()
            g = adv_latent.grad.data
            norm = g.abs().view(g.shape[0], -1).mean(dim=1, keepdim=True)
            g = g / norm.clamp(min=1e-12).view(-1,1,1)
            grad_m = momentum * grad_m + g
            adv_latent.data += alpha/8 * grad_m.sign()
            adv_latent.data = adv_latent.data
            adv_latent.grad.zero_()
        return adv_latent.detach()
    
#----------------------------------------------------

    def train(self):
        reid_transform = transforms.Compose([
            transforms.Resize((256, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        reidmodel = ft_net(751).to(global_config.device)
        save_path = os.path.join('../pretrained_model/resnet50_reid.pth')
        reidmodel.load_state_dict(torch.load(save_path), strict=False)
        reidmodel.classifier.classifier = nn.Sequential()
        reidmodel = reidmodel.eval()
        for p in reidmodel.parameters():
            p.requires_grad = False

        result = scipy.io.loadmat('../pretrained_model/vector_store.mat')
        gallery_e4e = torch.FloatTensor(result['gallery_e4e']).to(global_config.device)
        gallery_reid = torch.FloatTensor(result['gallery_f']).cpu().numpy()
        gallery_reid = np.ascontiguousarray(gallery_reid, dtype=np.float32)
        faiss.normalize_L2(gallery_reid)

#----------------------------------------------------
### RAG ###

        # Retrieval
        for _, (fname, image, img_path) in tqdm(enumerate(self.data_loader)):
            n, c, h, w = image.size()
            latent = torch.FloatTensor(n,14,512).zero_().cuda()
            img = Image.open(img_path[0]).convert('RGB')
            img = reid_transform(img)
            img = img.unsqueeze(0).to(global_config.device)
            with torch.no_grad():
                outputs = reidmodel(img)
            outputs = outputs.detach().cpu().numpy()
            outputs = np.ascontiguousarray(outputs, dtype=np.float32)
            faiss.normalize_L2(outputs)
            index = faiss.IndexFlatIP(512)
            index.add(gallery_reid)
            top_k = 10
            _, top_indices = index.search(outputs, top_k)
            selected_latent = gallery_e4e[top_indices].contiguous().clone()
            
#----------------------------------------------------

            # Augmentation
            n, c, d = selected_latent.shape
            Q = K = V = selected_latent.permute(1, 0, 2)
        
            trans_weights = torch.matmul(Q, K.transpose(2, 1)) / math.sqrt(d)
            inv_weights = 1.0 / trans_weights
            tmp = 0.1
            soft_weights = F.softmax(inv_weights/tmp, dim=-1)
            sum_weights = torch.matmul(soft_weights, V)
            weights = sum_weights.permute(1, 0, 2)
            least_similar = torch.mean(weights, dim=0, keepdim=True)
            latent[:] = least_similar

            image_name = fname[0]

            real_images_batch = image.to(global_config.device)
            with torch.no_grad():
                rimg = (resize(real_images_batch, size=(256, 128),
                               interpolation=InterpolationMode.BICUBIC) + 1) / 2
                imagenet_normalize = Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                rimg = imagenet_normalize(rimg)
                orig_feat = reidmodel(rimg)

#----------------------------------------------------

            # Generation
            sep = 4
            latent_coarse = latent[:, :sep, :].clone() 
            latent_fine = latent[:, sep:, :].clone() 

            latent_modified_coarse = self.MI_FGSM_l2(
                self.G,
                latent_coarse,
                latent_fine,
                real_images_batch,
                epsilon=hyperparameters.epsilon,
                alpha=hyperparameters.alpha,
                lambda_id=hyperparameters.lambda_id,
                momentum=hyperparameters.momentum,
                num_iter=hyperparameters.num_iter
            )

            latent_modified_fine = self.MI_FGSM(
                self.G,
                reidmodel,
                latent_modified_coarse,
                latent_fine,
                orig_feat,
                epsilon=hyperparameters.epsilon,
                alpha=hyperparameters.alpha,
                lambda_id=hyperparameters.lambda_id,
                momentum=hyperparameters.momentum,
                num_iter=hyperparameters.num_iter
            )

            final_latent = torch.cat([latent_modified_coarse, latent_modified_fine], dim=1)
            save_w(final_latent, self.G, image_name, '', './results', resize_to=(128, 64))