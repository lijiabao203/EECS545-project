# EECS545-project
This is my EECS 545 project file in University of Michigan.


- Cary Shu: For the DCPAN by Cary Shu, the base DCPAN splits three channels for height, width, and depth, with convolutions of (1,3,3) for depth, (3,1,3) for height, and (3,3,1) for width. The learning rate is 1e-3, output channels are 8, and 8 attention heads. Flash attention was added and found slightly better performance. More output channels could be used for a better feature mapping, but due to hardware memory limitations, 8 was the best amount chosen for our situation. Finally, after the feature map is produced, three 3d convolution layers of kernel size 3 with stride 3 are used to reduce the dimension of the output, followed by three fully connected three fully connected layers to the predicted age. Gradient accumulation is used with 4 accumulation steps, and the models were trained each for 5 epochs.
