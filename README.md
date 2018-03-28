# CT_denoise_use_DWConvResNet

This is a CT denoise model.
In reality, the projection data will have noise, so it need to design a 
model to denoise the final reconstruction result.

In our experiment, we add noise into the projection data, and then 
use FBP algorithm to reconstruction, after that use the DWConvResNet
to train by the reconstruction result with noise, compare with the
clean reconstruction result.

The DWConvResNet is base ResNet but we remove the pooling layer and change 
conv to depthwise conv(in tflearn, the depthwise conv is grouped_conv_2d).
We define a new resnet_dwconv_block like below:
     
    bn->relu->depthwiseconv->bn->relu->conv->bn->relu->depthwiseconv->bn->relu->conv

And the network basic structure is:

       resnet_dwconv_block(outputchannels=64)
       resnet_dwconv_block(outputchannels=128)
       resnet_dwconv_block(outputchannels=256)
       resnet_dwconv_block(outputchannels=128)
       resnet_dwconv_block(outputchannels=64)

The loss is begin with 7376, final loss is 20.77. Here is the loss curve:
![image](https://github.com/PaulGitt/CT_denoise_use_DWConvResNet/blob/master/loss.png)

This model can be trained by denosie the gaussian noise, the result like this:
The first one is the original picture, second one is add gaussian nosie, 
the third one is the reconstruction result by using DWConvResNet
![image](https://github.com/PaulGitt/CT_denoise_use_DWConvResNet/blob/master/result1.jpg)

And alse can be trained by denoise the multi-noise(like gaussian noise, poisson nosie and salt&pepper nosie),
result like below:
The first column is orignial picture,
second column is add noise(first one is add poisson noise, second one is add gaussian noise, third one is add
salt&pepper noise).
![image](https://github.com/PaulGitt/CT_denoise_use_DWConvResNet/blob/master/result2.jpg)
