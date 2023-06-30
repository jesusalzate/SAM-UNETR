from typing import Sequence, Tuple, Union

import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep
import torch.nn.functional as F
from .sam_image_encoder import ImageEncoderViT, LayerNorm2d, PatchEmbed
import torch
from functools import partial


class SAMUNETR(nn.Module):
    
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
        print('Using SAMUNETR_V2')

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(vit_patch_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if embed_dim % feature_size != 0:
            raise ValueError("embed_dim should be divisible by feature_size.")

        self.normalize = normalize

        # Image Encoder using Vision Transformer (ViT)
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
                
        
        
        # Encoder blocks        
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=embed_dim,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=True,
            res_block=True,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=embed_dim,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=True,
            res_block=True,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=embed_dim,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=True,
            res_block=True,
        )
        
        
        # Decoder blocks
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=embed_dim,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
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
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [embed_dim]


    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x
    
    def resize_tensor(self,tensor, output_shape):
        # check if the input tensor has a batch dimension
        has_batch_dim = tensor.dim() == 4 
        # add a batch dimension if needed
        if not has_batch_dim:
            tensor=tensor.view(1,*tensor.shape,1)
        # reshape the tensor to (batch_size, channel, *spatial)
        tensor = tensor.permute(0, -1, *range(1, len(tensor.shape) - 1))
        # resize the tensor using interpolate
        resized_tensor = F.interpolate(tensor, size=output_shape, mode='bilinear')
        
        resized_tensor=resized_tensor.permute(0,2,3,1)
        # remove the batch dimension if needed
        if not has_batch_dim:
            resized_tensor = resized_tensor.squeeze()

        return resized_tensor
    

    def forward(self, x_in):
        x, hidden_states_out = self.image_encoder_vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[7]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[15]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[23]
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(hidden_states_out[31])
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        return self.out(out)