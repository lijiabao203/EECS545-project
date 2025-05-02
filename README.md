# EECS545-project
This is my EECS 545 project file in University of Michigan.

- Cary Shu: 

For the DCPAN by Cary Shu, the base DCPAN splits three channels for height, width, and depth, with convolutions of (1,3,3) for depth, (3,1,3) for height, and (3,3,1) for width. The learning rate is 1e-3, output channels are 8, and 8 attention heads. Flash attention was added and found slightly better performance. More output channels could be used for a better feature mapping, but due to hardware memory limitations, 8 was the best amount chosen for our situation. Finally, after the feature map is produced, three 3d convolution layers of kernel size 3 with stride 3 are used to reduce the dimension of the output, followed by three fully connected three fully connected layers to the predicted age. Gradient accumulation is used with 4 accumulation steps, and the models were trained each for 5 epochs.

- Jiabao Li:

CNNs in baseline models/lr part: 3D AlexNet, 3D ResNet and 3D DenseNet

3D Vision Transformer: use flash attention in this module to accelerate execution. It uses a patch size of 32 and an embedding dimension of 512. The architecture includes 8 encoder layers, each with 8 attention heads. 

3D SWIN transformer: We use a patch size of 4 to divide the input images into non-overlapping patches, and set the embedding dimension to 96. The depths of each basic layer are (2, 2, 6, 2), indicating the number of transformer blocks in each stage. The number of attention heads per stage is set to (3, 6, 12, 24), and the window size is 7, which defines the local window used for computing self-attention.

DCPAN-SWIN transformer: After removing the fully connected layers from both DCPAN and the SWIN Transformer, we concatenated their feature vectors and used them as input to a more complex fully connected model for age prediction (channels are: feature dimension, 1024, 256, 1). The original hyperparameters of the backbone models were kept unchanged.


- Yuning Wang:
complete baseline models. The models are from https://github.com/Duplums/brain_age_with_site_removal. I used LR-finer to find the optimal learning rate. 

- YiMing Chen:

See recording branch readme.

- Reference:

See reference of https://www.overleaf.com/read/dxhvjxdpgtng#b6ae18
