# EECS545-project
This is my EECS 545 project file in University of Michigan.

Cary Shu:
- dcpan_full_data.log contains the log data for base DCPAN on the full dataset.
- dcpan_site1.log contains the log data for base DCPAN on just site 1 data.
- dcpan_flash_atten_full_data.log contains the log data for DCPAN with Flash Attention on the full dataset.
- dcpan_flash_atten_site1.log contains the log data for DCPAN with Flash Attention on just site 1 data.

Yiming Chen:
Remove the fully connected layers from both DCPAN and the SWIN Transformer, and concatenate their feature vectors and used them as input to a more complex fully connected model for age prediction (channels are: feature dimension, 1024, 256, 1).
Complete, test, and debug the models of SWIN and DCPAN-SWIN transformer, train it on GreatLake, record the loss, evaluate the MAE on the validation set, and perform testing on the final test set.
- Due to the time limitaion of Great Lake, the training process of DCPAN-SWIN transformer was divided into 3 parts.
- DCPAN-SWIN_epoch1-4.html contains the training log of epochs 1-4 for DCPAN-SWIN trnasformer on the full dataset.
- DCPAN-SWIN_epoch5-6.html contains the training log of epochs 5-6 for DCPAN-SWIN trnasformer on the full dataset.
- DCPAN-SWIN_epoch7-10.html contains the training log of epochs 7-10 for DCPAN-SWIN trnasformer on the full dataset.
- DCPAN-SWIN_prediction contains the prediction of test dataset in epoch 9 & 10, comparing with true data.
- SWIN-epoch1-10.html contains the log data of epochs 1-10 for SWIN on the full dataset.
