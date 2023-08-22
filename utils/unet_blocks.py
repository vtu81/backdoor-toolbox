"""
Class definitions for a standard U-Net Up-and Down-sampling blocks
http://arxiv.org/abs/1505.0.397
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """
    Instances the Encoder block that forms a part of a U-Net
    Parameters:
        in_channels (int): Depth (or number of channels) of the tensor that the block acts on
        filter_num (int) : Number of filters used in the convolution ops inside the block,
                             depth of the output of the enc block
        dropout(bool) : Flag to decide whether a dropout layer should be applied
        dropout_rate (float) : Probability of dropping a convolution output feature channel
    """
    def __init__(self, filter_num=64, in_channels=1, dropout=False, dropout_rate=0.3):

        super(EncoderBlock,self).__init__()
        self.filter_num = int(filter_num)
        self.in_channels = int(in_channels)
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.filter_num,
                               kernel_size=3,
                               padding=1)

        self.conv2 = nn.Conv2d(in_channels=self.filter_num,
                               out_channels=self.filter_num,
                               kernel_size=3,
                               padding=1)

        self.bn_op_1 = nn.InstanceNorm2d(num_features=self.filter_num, affine=True)
        self.bn_op_2 = nn.InstanceNorm2d(num_features=self.filter_num, affine=True)

        # Use Dropout ops as nn.Module instead of nn.functional definition
        # So using .train() and .eval() flags, can modify their behavior for MC-Dropout
        if dropout is True:
            self.dropout_1 = nn.Dropout(p=dropout_rate)
            self.dropout_2 = nn.Dropout(p=dropout_rate)

    def apply_manual_dropout_mask(self, x, seed):
        # Mask size : [Batch_size, Channels, Height, Width]
        dropout_mask = torch.bernoulli(input=torch.empty(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).fill_(self.dropout_rate),
                                       generator=torch.Generator().manual_seed(seed))

        x = x*dropout_mask.to(x.device)

        return x

    def forward(self, x, seeds=None):

        if seeds is not None:
            assert(seeds.shape[0] == 2)

        x = self.conv1(x)
        x = self.bn_op_1(x)
        x = F.leaky_relu(x)
        if self.dropout is True:
            if seeds is None:
                x = self.dropout_1(x)
            else:
                x = self.apply_manual_dropout_mask(x, seeds[0].item())

        x = self.conv2(x)
        x = self.bn_op_2(x)
        x = F.leaky_relu(x)
        if self.dropout is True:
            if seeds is None:
                x = self.dropout_2(x)
            else:
                x = self.apply_manual_dropout_mask(x, seeds[1].item())

        return x


class DecoderBlock(nn.Module):
    """
    Decoder block used in the U-Net
    Parameters:
        in_channels (int) : Number of channels of the incoming tensor for the upsampling op
        concat_layer_depth (int) : Number of channels to be concatenated via skip connections
        filter_num (int) : Number of filters used in convolution, the depth of the output of the dec block
        interpolate (bool) : Decides if upsampling needs to performed via interpolation or transposed convolution
        dropout(bool) : Flag to decide whether a dropout layer should be applied
        dropout_rate (float) : Probability of dropping a convolution output feature channel
    """
    def __init__(self, in_channels, concat_layer_depth, filter_num, interpolate=False, dropout=False, dropout_rate=0.3):

        # Up-sampling (interpolation or transposed conv) --> EncoderBlock
        super(DecoderBlock, self).__init__()
        self.filter_num = int(filter_num)
        self.in_channels = int(in_channels)
        self.concat_layer_depth = int(concat_layer_depth)
        self.interpolate = interpolate
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        # Upsample by interpolation followed by a 3x3 convolution to obtain desired depth
        self.up_sample_interpolate = nn.Sequential(nn.Upsample(scale_factor=2,
                                                               mode='bilinear',
                                                               align_corners=True),

                                                   nn.Conv2d(in_channels=self.in_channels,
                                                             out_channels=self.in_channels,
                                                             kernel_size=3,
                                                             padding=1)
                                                  )

        # Upsample via transposed convolution (know to produce artifacts)
        self.up_sample_tranposed = nn.ConvTranspose2d(in_channels=self.in_channels,
                                                      out_channels=self.in_channels,
                                                      kernel_size=3,
                                                      stride=2,
                                                      padding=1,
                                                      output_padding=1)

        self.down_sample = EncoderBlock(in_channels=self.in_channels+self.concat_layer_depth,
                                        filter_num=self.filter_num,
                                        dropout=self.dropout,
                                        dropout_rate=self.dropout_rate)

    def forward(self, x, skip_layer, seeds=None):
        if self.interpolate is True:
            up_sample_out = F.leaky_relu(self.up_sample_interpolate(x))
        else:
            up_sample_out = F.leaky_relu(self.up_sample_tranposed(x))

        merged_out = torch.cat([up_sample_out, skip_layer], dim=1)
        out = self.down_sample(merged_out, seeds=seeds)
        return out


class EncoderBlock3D(nn.Module):

    """
    Instances the 3D Encoder block that forms a part of a 3D U-Net
    Parameters:
        in_channels (int): Depth (or number of channels) of the tensor that the block acts on
        filter_num (int) : Number of filters used in the convolution ops inside the block,
                             depth of the output of the enc block
    """
    def __init__(self, filter_num=64, in_channels=1, dropout=False):

        super(EncoderBlock3D, self).__init__()
        self.filter_num = int(filter_num)
        self.in_channels = int(in_channels)
        self.dropout = dropout

        self.conv1 = nn.Conv3d(in_channels=self.in_channels,
                               out_channels=self.filter_num,
                               kernel_size=3,
                               padding=1)

        self.conv2 = nn.Conv3d(in_channels=self.filter_num,
                               out_channels=self.filter_num*2,
                               kernel_size=3,
                               padding=1)

        self.bn_op_1 = nn.InstanceNorm3d(num_features=self.filter_num)
        self.bn_op_2 = nn.InstanceNorm3d(num_features=self.filter_num*2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn_op_1(x)
        x = F.leaky_relu(x)
        if self.dropout is True:
            x = F.dropout3d(x, p=0.3)

        x = self.conv2(x)
        x = self.bn_op_2(x)
        x = F.leaky_relu(x)

        if self.dropout is True:
            x = F.dropout3d(x, p=0.3)

        return x


class DecoderBlock3D(nn.Module):
    """
    Decoder block used in the 3D U-Net
    Parameters:
        in_channels (int) : Number of channels of the incoming tensor for the upsampling op
        concat_layer_depth (int) : Number of channels to be concatenated via skip connections
        filter_num (int) : Number of filters used in convolution, the depth of the output of the dec block
        interpolate (bool) : Decides if upsampling needs to performed via interpolation or transposed convolution
    """
    def __init__(self, in_channels, concat_layer_depth, filter_num, interpolate=False, dropout=False):

        super(DecoderBlock3D, self).__init__()
        self.filter_num = int(filter_num)
        self.in_channels = int(in_channels)
        self.concat_layer_depth = int(concat_layer_depth)
        self.interpolate = interpolate
        self.dropout = dropout

        # Upsample by interpolation followed by a 3x3x3 convolution to obtain desired depth
        self.up_sample_interpolate = nn.Sequential(nn.Upsample(scale_factor=2,
                                                               mode='nearest'),

                                                  nn.Conv3d(in_channels=self.in_channels,
                                                            out_channels=self.in_channels,
                                                            kernel_size=3,
                                                            padding=1)
                                                 )

        # Upsample via transposed convolution (know to produce artifacts)
        self.up_sample_transposed = nn.ConvTranspose3d(in_channels=self.in_channels,
                                                       out_channels=self.in_channels,
                                                       kernel_size=3,
                                                       stride=2,
                                                       padding=1,
                                                       output_padding=1)

        if self.dropout is True:
            self.down_sample = nn.Sequential(nn.Conv3d(in_channels=self.in_channels+self.concat_layer_depth,
                                                       out_channels=self.filter_num,
                                                       kernel_size=3,
                                                       padding=1),

                                            nn.InstanceNorm3d(num_features=self.filter_num),

                                            nn.LeakyReLU(),

                                            nn.Dropout3d(p=0.3),

                                            nn.Conv3d(in_channels=self.filter_num,
                                                      out_channels=self.filter_num,
                                                      kernel_size=3,
                                                      padding=1),

                                            nn.InstanceNorm3d(num_features=self.filter_num),

                                            nn.LeakyReLU(),

                                            nn.Dropout3d(p=0.3))
        else:
            self.down_sample = nn.Sequential(nn.Conv3d(in_channels=self.in_channels+self.concat_layer_depth,
                                                       out_channels=self.filter_num,
                                                       kernel_size=3,
                                                       padding=1),

                                            nn.InstanceNorm3d(num_features=self.filter_num),

                                            nn.LeakyReLU(),

                                            nn.Conv3d(in_channels=self.filter_num,
                                                      out_channels=self.filter_num,
                                                      kernel_size=3,
                                                      padding=1),

                                            nn.InstanceNorm3d(num_features=self.filter_num),

                                            nn.LeakyReLU())

    def forward(self, x, skip_layer):

        if self.interpolate is True:
            up_sample_out = F.leaky_relu(self.up_sample_interpolate(x))
        else:
            up_sample_out = F.leaky_relu(self.up_sample_transposed(x))

        merged_out = torch.cat([up_sample_out, skip_layer], dim=1)
        out = self.down_sample(merged_out)
        return out