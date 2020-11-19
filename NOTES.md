
### Case 1
Train 2840 images = 2000 + 840
Test 210 images (all in Figaro)
Model: epoch-140.pth
=> IoU = 0.72


### Case 2
Pretrain MobileHairNet
Test: 210 images from Figaro
=> IoU = 0.7417

# Case 3
Train 352 epochs
Train 2840 images = 2000 + 840
Test 210 images (all in Figaro)
=> IoU = 0.7464

# Case 4
Train 328 epochs
Train images in Figaro1k
Test 210 images (all in Figaro)
=> IoU = 0.77

# Case 5
Train Lfw dataset
Test Lfw dataset (927)
Epoch = 315
=> IoU = 0.5885
