# CT_denoise_use_DWConvResNet

This is a CT denoise model.
Base ResNet but we remove the pooling layer and change conv to depthwise conv
(in tflearn, the depthwise conv is grouped_conv_2d).
We define a new resnet_dwconv_block like below:
      bn->relu->depthwiseconv->bn->relu->conv->bn->relu->depthwiseconv->bn->relu->conv
And the network basic structure is:
       resnet_dwconv_block(outputchannels=64)
       resnet_dwconv_block(outputchannels=128)
       resnet_dwconv_block(outputchannels=256)
       resnet_dwconv_block(outputchannels=128)
       resnet_dwconv_block(outputchannels=64)

The loss is begin with 7376, final loss is 20.77
