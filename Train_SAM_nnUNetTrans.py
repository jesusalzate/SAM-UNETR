#!/usr/bin/env python
# coding: utf-8




import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib

import monai
from monai.networks.nets import UNETR, UNet
from monai.utils import first
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, SmartCacheDataset
from monai.inferers import SliceInferer
import json
from time import sleep

from samunetr.SAMUNETR_V2 import SAMUNETR


fold =0
debug = False

data_dir = "/nvmescratch/ceib/Prostate/workdir/nnUNet_raw_data/Task2203_picai_baseline"

#Read folds and split data
json_path = "/nvmescratch/ceib/Prostate/workdir/splits/picai_nnunet/splits.json"
with open(json_path, "r") as f:
    splits = json.load(f)

fold_split = splits[fold]
train = fold_split['train']
val = fold_split['val']

train_images = []
train_labels = []

val_images = []
val_labels = []

for image_name in train:
    train_images.append([os.path.join(data_dir, "imagesTr", image_name + "_0000.nii.gz"),
                        os.path.join(data_dir, "imagesTr", image_name + "_0001.nii.gz"),
                        os.path.join(data_dir, "imagesTr", image_name + "_0002.nii.gz")])
    train_labels.append(os.path.join(data_dir, "labelsTr", image_name + ".nii.gz"))

for image_name in val:
    val_images.append([os.path.join(data_dir, "imagesTr", image_name + "_0000.nii.gz"),
                        os.path.join(data_dir, "imagesTr", image_name + "_0001.nii.gz"),
                        os.path.join(data_dir, "imagesTr", image_name + "_0002.nii.gz")])
    val_labels.append(os.path.join(data_dir, "labelsTr", image_name + ".nii.gz"))

print(f"Train: {len(train_images)} images")
print(f"Val: {len(val_images)} images")

#Organize Prostate158 Data
path='/nvmescratch/ceib/Prostate/input/prostate158/prostate158_train'
train_df_P158=pd.read_csv(os.path.join(path,'train.csv'))
test_df_P158=pd.read_csv(os.path.join(path,'valid.csv'))

train_df_P158=train_df_P158[train_df_P158['t2_tumor_reader1'].notna()]
test_df_P158=test_df_P158[test_df_P158['t2_tumor_reader1'].notna()]


columns = ['t2', 'adc', 'dwi','t2_anatomy_reader1', 'adc_tumor_reader1']

for df in [train_df_P158, test_df_P158]:
    for column in columns:
        df[column] = df[column].apply(lambda x: os.path.join(path,x))


# Flatten the lists for each modality
train_t2w = [img[0] for img in train_images]  # Extract all first images (T2-weighted) from each set
train_adc = [img[1] for img in train_images]  # Extract all second images (ADC) from each set
train_dwi = [img[2] for img in train_images]  # Extract all third images (DWI) from each set

val_t2w = [img[0] for img in val_images]  # Extract all first images (T2-weighted) from each set
val_adc = [img[1] for img in val_images]  # Extract all second images (ADC) from each set
val_dwi = [img[2] for img in val_images]  # Extract all third images (DWI) from each set

# Combine these flattened lists with the paths from the DataFrames
train_df = pd.DataFrame({
    't2w': train_t2w + list(train_df_P158['t2'].values),
    'adc': train_adc + list(train_df_P158['adc'].values),
    'dwi': train_dwi + list(train_df_P158['dwi'].values),
    'label': train_labels + list(train_df_P158['adc_tumor_reader1'].values)
})

test_df = pd.DataFrame({
    't2w': val_t2w + list(test_df_P158['t2'].values),
    'adc': val_adc + list(test_df_P158['adc'].values),
    'dwi': val_dwi + list(test_df_P158['dwi'].values),
    'label': val_labels + list(test_df_P158['adc_tumor_reader1'].values)
})


print(train_df.shape)
print(test_df.shape)

#breakpoint()

def Create_dataloaders(train_df,test_df,cache=False):

    """
    This function is for preprocessing, it contains only the basic transforms, but you can add more operations that you 
    find in the Monai documentation.
    https://monai.io/docs.html
    """

    #set_determinism(seed=0)

    img_columns=["t2","adc","dwi"]#,"adc","dwi"]
    label_column=["label"]
    
    mode=["bilinear","nearest"]#,"bilinear","bilinear","nearest"]#["bilinear","bilinear","bilinear", "nearest"]

    train_files = [{"t2": t2,'adc': adc,'dwi': dwi, "label": label} for 
                    t2,adc,dwi, label in zip(train_df['t2w'].values,
                                    train_df['adc'].values,
                                    train_df['dwi'].values,
                                    train_df['label'].values)]
    test_files = [{"t2": t2,'adc': adc,'dwi': dwi, "label": label} for 
                    t2,adc,dwi,label in zip(test_df['t2w'].values,
                                    test_df['adc'].values,
                                    test_df['dwi'].values,
                                    test_df['label'].values)]
    
    if debug:
        train_files=train_files[:10]
        test_files=test_files[:10]
    prob=0.175
    train_transforms = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=img_columns+label_column,reader="NibabelReader",image_only=True),
            monai.transforms.AsDiscreted(keys=label_column,threshold=1), #Convert values greater than 1 to 1
            monai.transforms.EnsureChannelFirstd(keys=img_columns+label_column),
            #monai.transforms.Resized(keys=img_columns+label_column,spatial_size=(128,128,-1),mode=("trilinear","trilinear","trilinear","nearest","nearest")),#SAMUNETR: Reshape to have the same dimension
            monai.transforms.ResampleToMatchd(keys=["adc","dwi","label"],key_dst="t2",mode=("bilinear","bilinear","nearest")),#Resample images to t2 dimension
            monai.transforms.ScaleIntensityd(keys=["t2","dwi"],minv=0.0, maxv=255.0),
            monai.transforms.ScaleIntensityRanged(
            keys=["adc"],
            a_min=315.09063720703125,
            a_max=2321.78369140625,
            b_min=0.0,
            b_max=255.0,
            clip=True,
            ),
            monai.transforms.NormalizeIntensityd(keys=img_columns,subtrahend=[114.495], divisor=[57.63],channel_wise=True),
            monai.transforms.ConcatItemsd(keys=img_columns, name='image', dim=0),
            monai.transforms.ConcatItemsd(keys=label_column, name='label', dim=0),
            monai.transforms.SpatialPadd(keys=["image", "label"], spatial_size=(128,128,-1)),
            monai.transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(128, 128, -1),
                pos=3,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            monai.transforms.RandRotated(
                keys=["image", "label"],
                prob=0.2,
                range_x=(-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
                range_y=(-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
                range_z=(-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
                mode=["bilinear", "nearest"],
            ),
            monai.transforms.RandScaleIntensityd(keys=["image"], prob=0.2, factors=(0.7, 1.4)),
            monai.transforms.RandGaussianNoised(keys="image", prob=0.1, mean=0, std=0.1),
            monai.transforms.RandGaussianSmoothd(keys="image", prob=0.1, sigma_x=(0.5, 1)),
            monai.transforms.RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.75, 1.25)),
            monai.transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0]),
            monai.transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[1]),
            monai.transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[2]),
            monai.transforms.ToTensord(keys=["image", "label"]),
            
        ]
    )
    test_transforms = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=img_columns+label_column,reader="NibabelReader",image_only=True),
            monai.transforms.AsDiscreted(keys=label_column,threshold=1), #Convert values greater than 1 to 1
            monai.transforms.EnsureChannelFirstd(keys=img_columns+label_column),
            #monai.transforms.Resized(keys=img_columns+label_column,spatial_size=(128,128,-1),mode=("trilinear","trilinear","trilinear","nearest","nearest")),#SAMUNETR: Reshape to have the same dimension
            monai.transforms.ResampleToMatchd(keys=["adc","dwi","label"],key_dst="t2",mode=("bilinear","bilinear","nearest")),#Resample images to t2 dimension
            monai.transforms.ScaleIntensityd(keys=["t2","dwi"],minv=0.0, maxv=255.0),
            monai.transforms.ScaleIntensityRanged(
            keys=["adc"],
            a_min=315.09063720703125,
            a_max=2321.78369140625,
            b_min=0.0,
            b_max=255.0,
            clip=True,
            ),
            monai.transforms.NormalizeIntensityd(keys=img_columns,subtrahend=[114.495], divisor=[57.63],channel_wise=True),
            monai.transforms.ConcatItemsd(keys=img_columns, name='image', dim=0),
            monai.transforms.ConcatItemsd(keys=label_column, name='label', dim=0),
            monai.transforms.SpatialPadd(keys=["image", "label"], spatial_size=(128,128,-1)),
            monai.transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    
    
    if cache:
        train_ds = SmartCacheDataset(
            data=train_files,
            transform=train_transforms,
            replace_rate=0.3,
            cache_num=700, 
            num_init_workers=4,
            num_replace_workers=4,
        )
        #train_ds = CacheDataset(data=train_files, transform=train_transforms,cache_rate=1.0,num_workers=8,copy_cache=False)#PerSlice(keys='image',transforms=train_transforms),cache_rate=1.0,num_workers=8,copy_cache=False)
        train_loader = DataLoader(train_ds, batch_size=1,shuffle=True)

        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0,num_workers=8,copy_cache=False)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        return train_loader,train_ds, test_loader,test_ds

    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=1,shuffle=True)
        test_ds = Dataset(data=test_files, transform=test_transforms)
        test_loader = DataLoader(test_ds, batch_size=1,shuffle=False)

        return train_loader,train_ds, test_loader,test_ds



def train2D(model, data_in, loss, optim, max_epochs, model_dir,device,name, test_interval=1):
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = []
    save_metric_test = []
    train_loader, test_loader = data_in

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0
        ap_metric_train=0
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_data in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                train_step += 1
                #To convert images to 2D
                volume_list = list(batch_data["image"])
                label_list = list(batch_data["label"])
                volume=torch.cat(volume_list,axis=-1)
                label=torch.cat(label_list,axis=-1)

                volume = monai.transforms.Transpose((3,0,1,2))(volume)
                label=monai.transforms.Transpose((3,0,1,2))(label)

                #######################################
                volume, labels = (volume.to(device), label.to(device))
                optim.zero_grad()
                outputs = model(volume)#[0]
                
                train_loss = loss(outputs, labels)

                train_loss.backward()
                optim.step()

                train_epoch_loss += train_loss.item()
            
                labels_list = decollate_batch(labels)
                labels_convert = [post_label(label_tensor) for label_tensor in labels_list]
                
                output_list = decollate_batch(outputs)
                output_convert = [post_pred(output_tensor) for output_tensor in output_list]
                
                dice_metric(y_pred=output_convert, y=labels_convert)
                iou_metric(y_pred=output_convert, y=labels_convert)
            
                tepoch.set_postfix(loss=train_loss.item(), dice_score=dice_metric.aggregate(reduction="mean").item())
                sleep(0.001)

            train_ds.update_cache()
            print('-'*20)

            train_epoch_loss /= train_step
            print(f'Epoch_loss: {train_epoch_loss:.4f}')
            save_loss_train.append(train_epoch_loss)
            np.save(os.path.join(model_dir, name+'_loss_train.npy'), save_loss_train)

            epoch_metric_train = dice_metric.aggregate(reduction="mean").item()
            dice_metric.reset()

            print(f'Epoch_metric: {epoch_metric_train:.4f}')
            
            iou_metric_train = iou_metric.aggregate(reduction="mean").item()
            iou_metric.reset()

            print(f'IoU_metric: {iou_metric_train:.4f}')
            

            save_metric_train.append(epoch_metric_train)
            np.save(os.path.join(model_dir, name+'_metric_train.npy'), save_metric_train)

            if (epoch) % test_interval == 0:

                model.eval()
                with torch.no_grad():
                    test_epoch_loss = 0
                    test_metric = 0
                    epoch_metric_test = 0
                    test_step = 0
                    ap_metric=0
                    for test_data in test_loader:
                        test_step += 1
                        test_volume, test_label = (test_data["image"].to(device),test_data["label"].to(device))
                        inferer=SliceInferer(roi_size=(128, 128),sw_batch_size=8,spatial_dim=2,progress=False, overlap=0.5)
                        test_outputs = inferer(test_volume, model)
                        
                        test_loss = loss(test_outputs, test_label)
                        test_epoch_loss += test_loss.item()
                        
                        labels_list = decollate_batch(test_label)
                        labels_convert = [post_label(label_tensor) for label_tensor in labels_list]

                        output_list = decollate_batch(test_outputs)
                        output_convert = [post_pred(output_tensor) for output_tensor in output_list]

                        dice_metric(y_pred=output_convert, y=labels_convert)
                        iou_metric(y_pred=output_convert, y=labels_convert)

                    test_epoch_loss /= test_step
                    print(f'test_loss_epoch: {test_epoch_loss:.4f}')
                    save_loss_test.append(test_epoch_loss)
                    np.save(os.path.join(model_dir, name+'_loss_test.npy'), save_loss_test)

                    epoch_metric_test=dice_metric.aggregate(reduction="mean").item()
                    
                    print(f'test_dice_epoch: {epoch_metric_test:.4f}')
                    print('test_dice_epoch_per_class:',dice_metric.aggregate())
                    
                    iou_metric_test=iou_metric.aggregate(reduction="mean").item()
                    
                    print(f'test_iou_epoch: {iou_metric_test:.4f}')
                    print('test_iou_epoch_per_class:',iou_metric.aggregate())
                    iou_metric.reset()
                    
                    save_metric_test.append(epoch_metric_test)
                    np.save(os.path.join(model_dir, name+'_metric_test.npy'), save_metric_test)
                    dice_metric.reset()
                    if epoch_metric_test > best_metric:
                        best_metric = epoch_metric_test
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), os.path.join(
                            model_dir, name+"_best_metric_model.pth"))

                    print(
                        f"current epoch: {epoch + 1} current mean dice: {epoch_metric_test:.4f}"
                        f"\nbest mean dice: {best_metric:.4f} "
                        f"at epoch: {best_metric_epoch}"
                    )


    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")




#Creating dataloaders

train_loader,train_ds,val_loader,val_ds=Create_dataloaders(train_df,test_df,cache=True)




pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Working on device: {device}')


# # Models

# ## Unet with Residual Units

model=SAMUNETR(img_size=128,in_channels=3,out_channels=2,trainable_encoder=True,pretrained=True).to(device)

# # Training Model


#loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250,2510860).to(device))
#loss_function = monai.losses.DiceCELoss(to_onehot_y=True, sigmoid=False,softmax=True,include_background=True)
loss_function = monai.losses.DiceFocalLoss(to_onehot_y=True, sigmoid=False,softmax=True,include_background=True)

torch.backends.cudnn.benchmark = True
optimizer = monai.optimizers.Novograd(model.parameters(), lr=0.001, weight_decay=0.01)





data_in=(train_loader,val_loader)
model_dir='/home/jaalzate/Tartaglia/Prostate_Tartaglia/Paper_Resultados/Results/SAMUnetr/Only_csPCa'



post_pred = monai.transforms.Compose([
        monai.transforms.AsDiscrete(argmax=True, to_onehot=2, num_classes=2),
        monai.transforms.KeepLargestConnectedComponent(),]
    )

post_label = monai.transforms.AsDiscrete(to_onehot=2)
dice_metric = monai.metrics.DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False,ignore_empty=True)
iou_metric=monai.metrics.MeanIoU(include_background=False,reduction="mean_batch",get_not_nans=False,ignore_empty=True)

train_ds.start()

train2D(model, data_in, loss_function, optimizer, 500, model_dir,device=device,name='SAMUnetrV2_128x128_pretrained_nnUNet_trans_fold0',test_interval=5)

train_ds.shutdown()