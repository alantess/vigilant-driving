# Vigilant Driving 
Deep Learning on the road.
### Requirements
- Requires [Pytorch](https://pytorch.org/) to run.
- Requires [Captum](https://captum.ai/) for model interpretability.

# Datasets
- [Segnet](https://bdd-data.berkeley.edu/)
- [Segnet_V2](https://github.com/commaai/comma10k)
- [Speed](https://github.com/commaai/speedchallenge/tree/master/data)
- [Depth](http://apolloscape.auto/stereo.html)

# SPEED
```
## Training Mode
- Use "-train" to set to training mode.
- Use "-epochs" to set the training epochs. Default is set to 1.
```sh
$ cd vigilant-driving/speed
$ mkdir models
$ python main.py -train True -epochs 1000
```
## Important notes
- For first time user please comment out the code below until a model has been saved. You cannot load a model that doesn't exist. Can be found in the train function.
```python
# model.load()
```
- In order to save the model uncomment the code below in the train function. 
```python
model.save()
```
- Downloading the dataset may be better than using links. If done, please set the directories of the videos and text file in the code.

# Directory Structure
------
    .
     ├── mobile             #  android app 
     ├── arm64              #  ARM64 support 
     ├── gui                #  Qt GUI 
     ├── Segnet             #  Segmentation on lanes
     ├── Segnet_V2          #  Complete segmentation on the environment
     ├── Depth              #  Segmentation on lanes
     ├── Speed              #  Speed Prediction
     ├── Etc                #  Random Files, Images, Gifs


# Visuals
|  ![](etc/original_driving_vid.gif) |  ![](etc/model_lanes.gif) |
|:-:|:-:|
|  ![](etc/depth_gif.gif) |![](etc/disparitygif.gif)   |
| <img src="etc/actual.jpg" alt="actual" width="200"/>  |  <img src="etc/interpret.jpg" alt="interpet" width="200"/> |
![](etc/segnet_v2.gif)



# Release 
- Available Models: ['disparitynet', 'segnet', 'ures', 'vidresnet', 'segnetv2']
- Please use the latest release to download models: 
    - alantess/vigilant-driving:main/{**latest_release**}
```python
import torch
repo = 'alantess/vigilant-driving:main/1.0.6'
model = torch.hub.load(repo, 'segnet', pretrained=True)
```
------


# To-Do
- [ ] Fix memory allocation in android 
- [ ] Complete trajectory analysis. 
- [ ] Deploy models on arm64 devices. 


# License
----

MIT
