# Hard No-Box Adversarial Attack on Skeleton-Based Human Action Recognition with Skeleton-Motion-Informed Gradient


### Requirements
```
 conda create -n  skeleton_contrast python=3.7 anaconda
 conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
 pip install tensorboard
 pip install rpy2
 pip install sklearn

```

### Data prepreprocessing instructions
*  Download raw  [NTU-RGB+D 60 ](https://github.com/shahroudy/NTURGB-D)  and save to ./data folder

```
cd data_gen
python ntu_gendata.py
```

### Pretrainning CL
python main_moco_inter_skeleton.py

### Get TV-AR(2) parameters(using R)
python AR.py

### Attack
python action_attackbyAR.py



