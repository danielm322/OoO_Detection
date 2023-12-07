#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().system('pip install albumentations==1.3.0')
get_ipython().system('pip install lightning-bolts==0.5.0')


# In[1]:


# from collections import namedtuple
import numpy as np
# import matplotlib.pyplot as plt
# import random
# import pandas as pd
# import seaborn as sns


# In[2]:


from icecream import ic
import torch
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from torchvision import transforms as transform_lib
# from pytorch_lightning.callbacks import TQDMProgressBar
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# 
# from dataset_utils.cityscapes import Cityscapes
# from dataset_utils.cityscapes import CityscapesDataModule
# from dataset_utils.woodscape import WoodScapeDataset
from dataset_utils.woodscape import WoodScapeDataModule
# from dataset_utils import WoodScapeSoilingDataset
from dataset_utils import WoodScapeSoilingDataModule

# from utils.display_images import denormalize_img
# from utils import show_dataset_image, show_dataset_mask
# from utils import show_prediction_images, show_prediction_uncertainty_images

from deeplab_v3p import DeepLabV3PlusModule
# from dropblock import DropBlock2D


# In[3]:


# from ls_ood_detect_cea.detectors import KDEClassifier, DetectorKDE
# from ls_ood_detect_cea.detectors import get_hz_scores
# from ls_ood_detect_cea.metrics import get_hz_detector_results
# # from ls_ood_detect_cea.metrics import get_ood_detector_results
# from ls_ood_detect_cea.metrics import plot_roc_ood_detector
# # from ls_ood_detect_cea.metrics import plot_auprc_ood_detector
# from ls_ood_detect_cea.dimensionality_reduction import plot_samples_pacmap
# from ls_ood_detect_cea.dimensionality_reduction import apply_pca_ds_split
# from ls_ood_detect_cea.dimensionality_reduction import apply_pca_transform

from ls_ood_detect_cea.uncertainty_estimation import Hook
from ls_ood_detect_cea.uncertainty_estimation import deeplabv3p_get_ls_mcd_samples, deeplabv3p_apply_dropout
from ls_ood_detect_cea.uncertainty_estimation import get_dl_h_z
# from ls_ood_detect_cea.ood_detection_dataset import build_ood_detection_ds
# from ls_ood_detect_cea.dimensionality_reduction import plot_samples_pacmap


# Steps for using the package:
# 
#     1. Load you Dataloader Pytorch-Lightning Module
#     2. Load your trained DNN PyTorch-Lightning Module
#     3. Add Hook to DNN Module for MC samples extraction
#     4. Get Monte-Carlo (MC) samples for In-Distribution (InD) samples dataloader, and Out-of-Distribution (OoD) samples dataloader
#     5. Get Entropy from InD and OoD MC samples
#     6. Build OoD Detection dataset (with InD and OoD samples)
#     7. Build OoD data-driven Detector (classifier)
#     8. Show OoD performance results
# 

# In[4]:


ws_dataset_path = './Woodscapes_dataset/'
batch_size = 1

cmap = {0: [0, 0, 0],  # "void"
        1: [128, 64, 128],  # "road",
        2: [69, 76, 11],  # "lanemarks",
        3: [0, 255, 0],  # "curb",
        4: [220, 20, 60],  # "person",
        5: [255, 0, 0],  # "rider",
        6: [0, 0, 142],  # "vehicles",
        7: [119, 11, 32],  # "bicycle",
        8: [0, 0, 230],  # "motorcycle",
        9: [220, 220, 0]  # "traffic_sign",
        }

# same values as in VainF Repository! - Probably not the best Values for Woodscapes!
ws_dlv3p_norm_mean = [0.485, 0.456, 0.406]
ws_dlv3p_norm_std = [0.229, 0.224, 0.225]


# In[5]:


ws_dm_normal_dlv3p = WoodScapeDataModule(dataset_dir=ws_dataset_path,
                                         img_size=(483, 640),
                                         batch_size=batch_size,
                                         default_transforms=True,
                                         label_colours=cmap,
                                         norm_mean=ws_dlv3p_norm_mean,
                                         norm_std=ws_dlv3p_norm_std,
                                         seed=9290,
                                         drop_last=True)
ws_dm_normal_dlv3p.setup()


# In[6]:


ws_dm_normal_dlv3p_256_512 = WoodScapeDataModule(dataset_dir=ws_dataset_path,
                                         img_size=(256, 512),
                                         batch_size=batch_size,
                                         default_transforms=True,
                                         label_colours=cmap,
                                         norm_mean=ws_dlv3p_norm_mean,
                                         norm_std=ws_dlv3p_norm_std,
                                         seed=9290,
                                         drop_last=True)
ws_dm_normal_dlv3p_256_512.setup()


# In[7]:


ws_dm_anomal_dlv3p = WoodScapeDataModule(dataset_dir=ws_dataset_path,
                                         img_size=(483, 640),
                                         batch_size=batch_size,
                                         default_transforms=True,
                                         label_colours=cmap,
                                         norm_mean=ws_dlv3p_norm_mean,
                                         norm_std=ws_dlv3p_norm_std,
                                         seed=9290,
                                         drop_last=True)
ws_dm_anomal_dlv3p.setup()


# In[8]:


ws_dlv3p_train_loader = ws_dm_normal_dlv3p.train_dataloader()
ws_dlv3p_valid_loader = ws_dm_normal_dlv3p.val_dataloader()
ws_dlv3p_test_loader = ws_dm_normal_dlv3p.test_dataloader()


# In[9]:


ws_dlv3p_anomaly_valid_loader = ws_dm_anomal_dlv3p.anomaly_val_dataloader()
ws_dlv3p_anomaly_test_loader = ws_dm_anomal_dlv3p.anomaly_test_dataloader()


# In[10]:


ws_256512_dlv3p_valid_loader = ws_dm_normal_dlv3p_256_512.val_dataloader()
ws_256512_dlv3p_test_loader = ws_dm_normal_dlv3p_256_512.test_dataloader()


# In[11]:


ic(len(ws_dlv3p_valid_loader));
ic(len(ws_dlv3p_test_loader));
ic(len(ws_dlv3p_anomaly_valid_loader));
ic(len(ws_dlv3p_anomaly_test_loader));
ic(len(ws_256512_dlv3p_valid_loader));
ic(len(ws_256512_dlv3p_test_loader));


# ## Woodscape soiling

# In[12]:


woodscape_soil_483640_dm = WoodScapeSoilingDataModule(dataset_dir="./Woodscapes_dataset/soiling_dataset/",
                                                      img_size=(483, 640),
                                                      batch_size=1,
                                                      default_transforms=True,
                                                      seed=9290)
woodscape_soil_483640_dm.setup()


# In[13]:


woodscape_soil_256512_dm = WoodScapeSoilingDataModule(dataset_dir="./Woodscapes_dataset/soiling_dataset/",
                                                      img_size=(256, 512),
                                                      batch_size=1,
                                                      default_transforms=True,
                                                      seed=9290)
woodscape_soil_256512_dm.setup()


# In[14]:


ws_soiling_483640_valid_loader = woodscape_soil_483640_dm.val_dataloader()
ws_soiling_483640_test_loader = woodscape_soil_483640_dm.test_dataloader()


# In[15]:


ws_soiling_256512_valid_loader = woodscape_soil_256512_dm.val_dataloader()
ws_soiling_256512_test_loader = woodscape_soil_256512_dm.test_dataloader()


# ## Deeplabv3+ Woodscape Trained Model

# In[16]:


ws_dlv3p_path = "./checkpoints/last.ckpt"
ws_dlv3p_model = DeepLabV3PlusModule.load_from_checkpoint(checkpoint_path=ws_dlv3p_path)


# In[17]:


ic(ws_dlv3p_model.pred_loss_type);
ic(ws_dlv3p_model.n_class);


# ## Add Hook Deeplabv3+ Woodscape

# In[18]:


ic(ws_dlv3p_model.deeplab_v3plus_model.drop_block1);
ic(ws_dlv3p_model.deeplab_v3plus_model.drop_block1.drop_prob);
ic(ws_dlv3p_model.deeplab_v3plus_model.drop_block1.training);


# In[19]:


ws_dlv3p_hook_dropblock2d_layer = Hook(ws_dlv3p_model.deeplab_v3plus_model.drop_block1)


# ## Get Monte Carlo Samples

# In[20]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[21]:


mc_samples = 32


# In[22]:


ws_dlv3p_model.deeplab_v3plus_model.to(device);
ws_dlv3p_model.deeplab_v3plus_model.eval(); 
ws_dlv3p_model.deeplab_v3plus_model.apply(deeplabv3p_apply_dropout); # enable dropout


# In[23]:


ic(ws_dlv3p_model.deeplab_v3plus_model.drop_block1.drop_prob);
ic(ws_dlv3p_model.deeplab_v3plus_model.drop_block1.training);


# In[24]:


ws_dlv3p_ws_normal_valid_32mc_samples = deeplabv3p_get_ls_mcd_samples(ws_dlv3p_model,
                                                                      ws_dlv3p_valid_loader,
                                                                      mc_samples,
                                                                      ws_dlv3p_hook_dropblock2d_layer)

ws_dlv3p_ws_normal_test_32mc_samples = deeplabv3p_get_ls_mcd_samples(ws_dlv3p_model,
                                                                     ws_dlv3p_test_loader,
                                                                     mc_samples,
                                                                     ws_dlv3p_hook_dropblock2d_layer)


# In[ ]:


ws_dlv3p_ws_anomal_valid_32mc_samples = deeplabv3p_get_ls_mcd_samples(ws_dlv3p_model,
                                                                      ws_dlv3p_anomaly_valid_loader,
                                                                      mc_samples,
                                                                      ws_dlv3p_hook_dropblock2d_layer)

ws_dlv3p_ws_anomal_test_32mc_samples = deeplabv3p_get_ls_mcd_samples(ws_dlv3p_model,
                                                                     ws_dlv3p_anomaly_test_loader,
                                                                     mc_samples,
                                                                     ws_dlv3p_hook_dropblock2d_layer)


# In[ ]:


# ws_dlv3p_cs_483640_valid_32mc_samples = deeplabv3p_get_ls_mcd_samples(ws_dlv3p_model,
#                                                                       cs_483640_dlv3p_valid_loader,
#                                                                       mc_samples,
#                                                                       ws_dlv3p_hook_dropblock2d_layer)
# 
# ws_dlv3p_cs_483640_test_32mc_samples = deeplabv3p_get_ls_mcd_samples(ws_dlv3p_model,
#                                                                      cs_483640_dlv3p_test_loader,
#                                                                      mc_samples,
#                                                                      ws_dlv3p_hook_dropblock2d_layer)


# In[ ]:


ws_dlv3p_ws_soiling_483640_valid_32mc_samples = deeplabv3p_get_ls_mcd_samples(ws_dlv3p_model,
                                                                              ws_soiling_483640_valid_loader,
                                                                              mc_samples,
                                                                              ws_dlv3p_hook_dropblock2d_layer)

ws_dlv3p_ws_soiling_483640_test_32mc_samples = deeplabv3p_get_ls_mcd_samples(ws_dlv3p_model,
                                                                             ws_soiling_483640_test_loader,
                                                                             mc_samples,
                                                                             ws_dlv3p_hook_dropblock2d_layer)


# In[ ]:


torch.save(ws_dlv3p_ws_normal_valid_32mc_samples,
           './mc_samples/ws_dlv3p_ws_normal_valid_32mc_samples.pt')
torch.save(ws_dlv3p_ws_normal_test_32mc_samples,
           './mc_samples/ws_dlv3p_ws_normal_test_32mc_samples.pt')


# In[ ]:


torch.save(ws_dlv3p_ws_anomal_valid_32mc_samples,
           './mc_samples/ws_dlv3p_ws_anomal_valid_32mc_samples.pt')
torch.save(ws_dlv3p_ws_anomal_test_32mc_samples,
           './mc_samples/ws_dlv3p_ws_anomal_test_32mc_samples.pt')


# In[ ]:


# torch.save(ws_dlv3p_cs_483640_valid_32mc_samples,
#            './mc_samples/ws_dlv3p_cs_483640_valid_32mc_samples.pt')
# torch.save(ws_dlv3p_cs_483640_test_32mc_samples,
#            './mc_samples/ws_dlv3p_cs_483640_test_32mc_samples.pt')


# In[ ]:


torch.save(ws_dlv3p_ws_soiling_483640_valid_32mc_samples,
           './mc_samples/ws_dlv3p_ws_soiling_483640_valid_32mc_samples.pt')
torch.save(ws_dlv3p_ws_soiling_483640_test_32mc_samples,
           './mc_samples/ws_dlv3p_ws_soiling_483640_test_32mc_samples.pt')


# In[ ]:


ws_dlv3p_ws_normal_valid_32mc_samples = torch.load('./mc_samples/ws_dlv3p_ws_normal_valid_32mc_samples.pt')
ws_dlv3p_ws_normal_test_32mc_samples = torch.load('./mc_samples/ws_dlv3p_ws_normal_test_32mc_samples.pt')

ws_dlv3p_ws_anomal_valid_32mc_samples = torch.load('./mc_samples/ws_dlv3p_ws_anomal_valid_32mc_samples.pt')
ws_dlv3p_ws_anomal_test_32mc_samples = torch.load('./mc_samples/ws_dlv3p_ws_anomal_test_32mc_samples.pt')

# ws_dlv3p_cs_483640_valid_32mc_samples = torch.load('./mc_samples/ws_dlv3p_cs_483640_valid_32mc_samples.pt')
# ws_dlv3p_cs_483640_test_32mc_samples = torch.load('./mc_samples/ws_dlv3p_cs_483640_test_32mc_samples.pt')

ws_dlv3p_ws_soiling_483640_valid_32mc_samples = torch.load('./mc_samples/ws_dlv3p_ws_soiling_483640_valid_32mc_samples.pt')
ws_dlv3p_ws_soiling_483640_test_32mc_samples = torch.load('./mc_samples/ws_dlv3p_ws_soiling_483640_test_32mc_samples.pt')


# ## Get entropy

# In[ ]:


_, ws_dlv3p_h_z_ws_normal_valid_samples_np = get_dl_h_z(ws_dlv3p_ws_normal_valid_32mc_samples, mcd_samples_nro=mc_samples)
_, ws_dlv3p_h_z_ws_normal_test_samples_np = get_dl_h_z(ws_dlv3p_ws_normal_test_32mc_samples, mcd_samples_nro=mc_samples)
_, ws_dlv3p_h_z_ws_anomal_valid_samples_np = get_dl_h_z(ws_dlv3p_ws_anomal_valid_32mc_samples, mcd_samples_nro=mc_samples)
_, ws_dlv3p_h_z_ws_anomal_test_samples_np = get_dl_h_z(ws_dlv3p_ws_anomal_test_32mc_samples, mcd_samples_nro=mc_samples)
# _, ws_dlv3p_h_z_cs_483640_valid_samples_np = get_dl_h_z(ws_dlv3p_cs_483640_valid_32mc_samples, mcd_samples_nro=32)
# _, ws_dlv3p_h_z_cs_483640_test_samples_np = get_dl_h_z(ws_dlv3p_cs_483640_test_32mc_samples, mcd_samples_nro=32)
_, ws_dlv3p_h_z_ws_soil_483640_valid_samples_np = get_dl_h_z(ws_dlv3p_ws_soiling_483640_valid_32mc_samples, mcd_samples_nro=mc_samples)
_, ws_dlv3p_h_z_ws_soil_483640_test_samples_np = get_dl_h_z(ws_dlv3p_ws_soiling_483640_test_32mc_samples, mcd_samples_nro=mc_samples)


# In[ ]:


np.save('./entropy/ws_dlv3p_h_z_ws_normal_valid_samples_np',
        ws_dlv3p_h_z_ws_normal_valid_samples_np)

np.save('./entropy/ws_dlv3p_h_z_ws_normal_test_samples_np',
        ws_dlv3p_h_z_ws_normal_test_samples_np)

np.save('./entropy/ws_dlv3p_h_z_ws_anomal_valid_samples_np',
        ws_dlv3p_h_z_ws_anomal_valid_samples_np)

np.save('./entropy/ws_dlv3p_h_z_ws_anomal_test_samples_np',
        ws_dlv3p_h_z_ws_anomal_test_samples_np)

# np.save('./entropy/ws_dlv3p_h_z_cs_483640_valid_samples_np',
#         ws_dlv3p_h_z_cs_483640_valid_samples_np)
# 
# np.save('./entropy/ws_dlv3p_h_z_cs_483640_test_samples_np',
#         ws_dlv3p_h_z_cs_483640_test_samples_np)

np.save('./entropy/ws_dlv3p_h_z_ws_soil_483640_valid_samples_np',
        ws_dlv3p_h_z_ws_soil_483640_valid_samples_np)

np.save('./entropy/ws_dlv3p_h_z_ws_soil_483640_test_samples_np',
        ws_dlv3p_h_z_ws_soil_483640_test_samples_np)

