# Vigilant Driving 
Deep Learning on the road.
### Requirements
- Requires [Pytorch](https://pytorch.org/) to run.
- Requires [Captum](https://captum.ai/) for model interpretability.

# Depth
 - Model: U-Net with a Resnet50 backbone
 - [Dataset](http://apolloscape.auto/stereo.html)
 
# SEGNET 
## Test mode:
- Run main.py. **MUST** have a saved model ready.
```sh
$ cd vigilant-driving/segnet
$ python main.py 
```
## Training Mode
- Use "-train" to set to training mode.
- Use "-epochs" to set the training epochs. Default is set to 3.
```sh
$ cd vigilant-driving/segnet
$ mkdir models
$ python main.py -train True -epochs 10
```
### Dataset
Drivable maps dataset can be found [here](https://bdd-data.berkeley.edu/)

# SPEED
- Dataset needed can be found [here](https://github.com/commaai/speedchallenge/tree/master/data)
## Test mode:
- Run main.py. **MUST** have a saved model ready.
```sh
$ cd vigilant-driving/speed
$ python main.py 
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
    torchvision.transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ├── Deoth              # Segmentation on lanes
        ├── dataset.py      #  Class to hold for the dataset
        ├── main.py         #  Main controller
        ├── network.py      # U Net 
        ├── visual.py        #  Test and visualize the model
    └── Speed               #  Speed Prediction
        ├── main.py         #  Main controller
        ├──test_pred.txt    #  Prediction for the training set
        ├──train_pred.txt   #  Prediction on the testing set
    └── Etc               #  Random Files, Images, Gifs


# Visuals
### Lanes 
|  LANES | COLOR   |
|:-:|---|
| Direct  | 🔴  |
| Alternative  |  🟢 |

![](etc/original_driving_vid.gif)
![](etc/model_lanes.gif)
### Depth
![](etc/depth_gif.gif)
### Model Interpretability 
<img src="etc/actual.jpg" alt="actual" width="600"/>
<img src="etc/interpret.jpg" alt="interpet" width="600"/>

# Release 
- [Download](https://github.com/alantess/vigilant-driving/releases) Models
------

# License
----

MIT
