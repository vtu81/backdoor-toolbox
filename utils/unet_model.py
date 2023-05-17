"""
A PyTorch Implementation of a U-Net.
Supports 2D (https://arxiv.org/abs/1505.04597) and 3D(https://arxiv.org/abs/1606.06650) variants
Author: Ishaan Bhat
Email: ishaan@isi.uu.nl
"""
from utils.unet_blocks import *
from math import pow


class UNet(nn.Module):
    """
     PyTorch class definition for the U-Net architecture for image segmentation
     Parameters:
         n_channels (int) : Number of image channels
         base_filter_num (int) : Number of filters for the first convolution (doubled for every subsequent block)
         num_blocks (int) : Number of encoder/decoder blocks
         num_classes(int) : Number of classes that need to be segmented
         mode (str): 2D or 3D
         use_pooling (bool): Set to 'True' to use MaxPool as downnsampling op.
                             If 'False', strided convolution would be used to downsample feature maps (http://arxiv.org/abs/1908.02182)
         dropout (bool) : Whether dropout should be added to central encoder and decoder blocks (eg: BayesianSegNet)
         dropout_rate (float) : Dropout probability
     Returns:
         out (torch.Tensor) : Prediction of the segmentation map
     """
    def __init__(self, n_channels=1, base_filter_num=64, num_blocks=4, num_classes=5, mode='2D', dropout=False, dropout_rate=0.3, use_pooling=True):

        super(UNet, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.expanding_path = nn.ModuleList()
        self.downsampling_ops = nn.ModuleList()

        self.num_blocks = num_blocks
        self.n_channels = int(n_channels)
        self.n_classes = int(num_classes)
        self.base_filter_num = int(base_filter_num)
        self.enc_layer_depths = []  # Keep track of the output depths of each encoder block
        self.mode = mode
        self.pooling = use_pooling
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        if mode == '2D':
            self.encoder = EncoderBlock
            self.decoder = DecoderBlock
            self.pool = nn.MaxPool2d

        elif mode == '3D':
            self.encoder = EncoderBlock3D
            self.decoder = DecoderBlock3D
            self.pool = nn.MaxPool3d
        else:
            print('{} mode is invalid'.format(mode))

        for block_id in range(num_blocks):
            # Due to GPU mem constraints, we cap the filter depth at 512
            enc_block_filter_num = min(int(pow(2, block_id)*self.base_filter_num), 512)  # Output depth of current encoder stage of the 2-D variant
            if block_id == 0:
                enc_in_channels = self.n_channels
            else:
                if self.mode == '2D':
                    if int(pow(2, block_id)*self.base_filter_num) <= 512:
                        enc_in_channels = enc_block_filter_num//2
                    else:
                        enc_in_channels = 512
                else:
                    enc_in_channels = enc_block_filter_num  # In the 3D UNet arch, the encoder features double in the 2nd convolution op


            # Dropout only applied to central encoder blocks -- See BayesianSegNet by Kendall et al.
            if self.dropout is True and block_id >= num_blocks-2:
                self.contracting_path.append(self.encoder(in_channels=enc_in_channels,
                                                          filter_num=enc_block_filter_num,
                                                          dropout=True,
                                                          dropout_rate=self.dropout_rate))
            else:
                self.contracting_path.append(self.encoder(in_channels=enc_in_channels,
                                                          filter_num=enc_block_filter_num,
                                                          dropout=False))
            if self.mode == '2D':
                self.enc_layer_depths.append(enc_block_filter_num)
                if self.pooling is False:
                    self.downsampling_ops.append(nn.Sequential(nn.Conv2d(in_channels=self.enc_layer_depths[-1],
                                                                         out_channels=self.enc_layer_depths[-1],
                                                                         kernel_size=3,
                                                                         stride=2,
                                                                         padding=1),
                                                                nn.InstanceNorm2d(num_features=self.filter_num),
                                                                nn.LeakyReLU()))
            else:
                self.enc_layer_depths.append(enc_block_filter_num*2) # Specific to 3D U-Net architecture (due to doubling of #feature_maps inside the 3-D Encoder)
                if self.pooling is False:
                    self.downsampling_ops.append(nn.Sequential(nn.Conv3d(in_channels=self.enc_layer_depths[-1],
                                                                         out_channels=self.enc_layer_depths[-1],
                                                                         kernel_size=3,
                                                                         stride=2,
                                                                         padding=1),
                                                                nn.InstanceNorm3d(num_features=self.enc_layer_depths[-1]),
                                                                nn.LeakyReLU()))

        # Bottleneck layer
        if self.mode == '2D':
            bottle_neck_filter_num = self.enc_layer_depths[-1]*2
            bottle_neck_in_channels = self.enc_layer_depths[-1]
            self.bottle_neck_layer = self.encoder(filter_num=bottle_neck_filter_num,
                                                  in_channels=bottle_neck_in_channels)

        else:  # Modified for the 3D UNet architecture
            bottle_neck_in_channels = self.enc_layer_depths[-1]
            bottle_neck_filter_num = self.enc_layer_depths[-1]*2
            self.bottle_neck_layer =  nn.Sequential(nn.Conv3d(in_channels=bottle_neck_in_channels,
                                                              out_channels=bottle_neck_in_channels,
                                                              kernel_size=3,
                                                              padding=1),

                                                    nn.InstanceNorm3d(num_features=bottle_neck_in_channels),

                                                    nn.LeakyReLU(),

                                                    nn.Conv3d(in_channels=bottle_neck_in_channels,
                                                              out_channels=bottle_neck_filter_num,
                                                              kernel_size=3,
                                                              padding=1),

                                                    nn.InstanceNorm3d(num_features=bottle_neck_filter_num),

                                                    nn.LeakyReLU())

        # Decoder Path
        dec_in_channels = int(bottle_neck_filter_num)
        for block_id in range(num_blocks):
            if self.dropout is True and block_id < 2:
                self.expanding_path.append(self.decoder(in_channels=dec_in_channels,
                                                        filter_num=self.enc_layer_depths[-1-block_id],
                                                        concat_layer_depth=self.enc_layer_depths[-1-block_id],
                                                        interpolate=False,
                                                        dropout=True,
                                                        dropout_rate=self.dropout_rate))
            else:
                self.expanding_path.append(self.decoder(in_channels=dec_in_channels,
                                                        filter_num=self.enc_layer_depths[-1-block_id],
                                                        concat_layer_depth=self.enc_layer_depths[-1-block_id],
                                                        interpolate=False,
                                                        dropout=False))

            dec_in_channels = self.enc_layer_depths[-1-block_id]

        # Output Layer
        if mode == '2D':
            self.output = nn.Conv2d(in_channels=int(self.enc_layer_depths[0]),
                                    out_channels=self.n_classes,
                                    kernel_size=1)
        else:
            self.output = nn.Conv3d(in_channels=int(self.enc_layer_depths[0]),
                                    out_channels=self.n_classes,
                                    kernel_size=1)

    def forward(self, x, seeds=None):

        if self.mode == '2D':
            h, w = x.shape[-2:]
        else:
            d, h, w = x.shape[-3:]

        # Encoder
        enc_outputs = []
        seed_index = 0
        for stage, enc_op in enumerate(self.contracting_path):
            if stage >= len(self.contracting_path) - 2:
                if seeds is not None:
                    x = enc_op(x, seeds[seed_index:seed_index+2])
                else:
                    x = enc_op(x)
                seed_index += 2 # 2 seeds required per block
            else:
                x = enc_op(x)
            enc_outputs.append(x)

            if self.pooling is True:
                x = self.pool(kernel_size=2)(x)
            else:
                x = self.downsampling_ops[stage](x)

        # Bottle-neck layer
        x = self.bottle_neck_layer(x)
        # Decoder
        for block_id, dec_op in enumerate(self.expanding_path):
            if block_id < 2:
                if seeds is not None:
                    x = dec_op(x, enc_outputs[-1-block_id], seeds[seed_index:seed_index+2])
                else:
                    x = dec_op(x, enc_outputs[-1-block_id])
                seed_index += 2
            else:
                x = dec_op(x, enc_outputs[-1-block_id])


        # Output
        x = self.output(x)

        return x