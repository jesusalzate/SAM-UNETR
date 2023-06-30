from typing import Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep
from .sam_image_encoder import ImageEncoderViT, LayerNorm2d, PatchEmbed
from functools import partial
import torch.nn.functional as F




class SAMUNETR(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """
    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        feature_size: int = 20,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        spatial_dims: int = 2,
        embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        vit_patch_size = 16,
        encoder_out_channels = 256,
        pretrained=True,
        trainable_encoder=True
        
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            spatial_dims: number of spatial dims.
            embed_dim: embeding dimensions for sam encoder.
            encoder_depth= number of attention blocks for the encoder
            encoder_num_heads: number of attention heads.
            encoder_global_attn_indexes: Indexes for blocks using global attention.
            vit_patch_size: Patch size
            encoder_out_channels: number of output channels for the encoder.
            pretrained: Wheter to use pretrained model or not
        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if embed_dim % feature_size != 0:
            raise ValueError("embed_dim should be divisible by feature_size.")

        self.normalize = normalize

        
        self.image_encoder_vit=ImageEncoderViT(
                    depth=encoder_depth,
                    embed_dim=embed_dim,
                    img_size=img_size[0],
                    mlp_ratio=4,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                    num_heads=encoder_num_heads,
                    patch_size=vit_patch_size,
                    qkv_bias=True,
                    use_rel_pos=True,
                    global_attn_indexes=encoder_global_attn_indexes,
                    window_size=14,
                    out_chans=encoder_out_channels,
                    in_chans=in_channels
        )#.to('cuda')
        
        if pretrained:
            model_state=torch.load('./models_SAM/sam_image_encoder.pth')

            if img_size!=1024:
            
                #Reshape model weights to match with new image size
                new_pos_embed_size=(img_size[0]//16, img_size[0]//16)
                new_att_block_size=((img_size[0]//8)-1, embed_dim//16)
                
                if in_channels==1:
                    model_state['patch_embed.proj.weight']=model_state['patch_embed.proj.weight'].mean(axis=1).unsqueeze(1) #Convert from RGB input to grayscale input
                
                elif in_channels!=3:
                    model_state['patch_embed']=PatchEmbed(
                                                kernel_size=(vit_patch_size, vit_patch_size),
                                                stride=(vit_patch_size, vit_patch_size),
                                                in_chans=in_channels,
                                                embed_dim=embed_dim,
                                                )
                    model_state['patch_embed.proj.weight']=F.interpolate(model_state['patch_embed.proj.weight'].view(1,*model_state['patch_embed.proj.weight'].shape),
                                                                         size=(in_channels,vit_patch_size,vit_patch_size), mode='trilinear').squeeze()
                
                print(model_state['patch_embed.proj.weight'].shape)
                unmatched_layers = {
                                        'pos_embed': new_pos_embed_size,
                                        'blocks.7.attn.rel_pos_h': new_att_block_size,
                                        'blocks.7.attn.rel_pos_w': new_att_block_size,
                                        'blocks.15.attn.rel_pos_h': new_att_block_size,
                                        'blocks.15.attn.rel_pos_w': new_att_block_size,
                                        'blocks.23.attn.rel_pos_h': new_att_block_size,
                                        'blocks.23.attn.rel_pos_w': new_att_block_size,
                                        'blocks.31.attn.rel_pos_h': new_att_block_size,
                                        'blocks.31.attn.rel_pos_w': new_att_block_size
                                    }
                for key,size in unmatched_layers.items():
                    model_state[key]=self.resize_tensor(model_state[key], size)
        
            self.image_encoder_vit.load_state_dict(state_dict=model_state)
            
            if not trainable_encoder:
                for param in self.image_encoder_vit.parameters():
                    param.requires_grad = False
                print('Image encoder no trainable')

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=embed_dim,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=embed_dim,
            out_channels=feature_size*2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=embed_dim,
            out_channels=feature_size*4,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=encoder_out_channels,
            out_channels=16*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*16,
            out_channels=feature_size*8,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        
        
        
    def upsample_block(self,data,in_channels,out_channels,scale):
        block=nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale, stride=scale),#.to('cuda')
            LayerNorm2d(out_channels)
        ).to('cuda')
        
        return block(data)
    
    
    def resize_tensor(self,tensor, output_shape):
        # check if the input tensor has a batch dimension
        has_batch_dim = tensor.dim() >= 4 
        # add a batch dimension if needed
        if not has_batch_dim:
            tensor=tensor.view(1,*tensor.shape,1)
        # reshape the tensor to (batch_size, channel, *spatial)
        tensor = tensor.permute(0, -1, *range(1, len(tensor.shape) - 1))
        #print(tensor.shape)
        # resize the tensor using interpolate
        
        resized_tensor = F.interpolate(tensor, size=output_shape, mode='bilinear')
        
        resized_tensor=resized_tensor.permute(0,2,3,1)
        # remove the batch dimension if needed
        if not has_batch_dim:
            resized_tensor = resized_tensor.squeeze()

        return resized_tensor
    
    def forward(self, x_in):
        x,hidden_states_out = self.image_encoder_vit(x_in)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[7].permute(0,3,1,2))
        enc2 = self.encoder3(hidden_states_out[15].permute(0,3,1,2))
        enc3 = self.encoder4(hidden_states_out[23].permute(0,3,1,2))
        dec4 = self.encoder10(x)

        
    
        
        up_sampled_hs=SAMUNETR.upsample_block(self,data=hidden_states_out[31].permute(0,3,1,2), 
                                              in_channels=hidden_states_out[31].permute(0,3,1,2).shape[1],
                                              out_channels=(dec4.shape[1])//2,scale=1)
        

        dec3 = self.decoder5(dec4,up_sampled_hs)

        up_sampled_enc3=SAMUNETR.upsample_block(self,data=enc3, 
                                              in_channels=enc3.shape[1],
                                              out_channels=enc3.shape[1],scale=2)
        dec2 = self.decoder4(dec3, up_sampled_enc3)

        
        up_sampled_enc2=SAMUNETR.upsample_block(self,data=enc2, 
                                              in_channels=enc2.shape[1],
                                              out_channels=enc2.shape[1],scale=4)
        dec1 = self.decoder3(dec2, up_sampled_enc2)

        up_sampled_enc1=SAMUNETR.upsample_block(self,data=enc1, 
                                              in_channels=enc1.shape[1],
                                              out_channels=enc1.shape[1],scale=8)
        dec0 = self.decoder2(dec1, up_sampled_enc1)

#         up_sampled_enc0=SAMUNETR.upsample_block(self,data=enc0, 
#                                               in_channels=enc0.shape[1],
#                                               out_channels=enc0.shape[1],scale=2)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
   
        return logits