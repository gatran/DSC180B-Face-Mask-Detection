# DSC180B Face Mask Detection

This repository focuses on the creation of a Face Mask Detection (website/app/report)

-----------------------------------------------------------------------------------------------------------------


### Introduction
* This repo is about training an Convolutional Neunral Network(CNN) image classification model on MaskedFace-Net. MaskedFace-Net is a dataset that contains more than 60,000 images of person either wearing a mask not. For images that contain a person wearing a mask, the dataset is further splited into either a person is wearing a mask properly or not. In this repo, we've trained a model on this dataset and also implemented a Grad-CAM algorithm on the model.

##### config 
* This folder contains the parameters for running the target.

##### model
* This folder contains a trained model parameters(model.pt)

##### notebook
* This folder contains the exploratory data analysis(EDA) of the MaskedFace-Net.

##### result
* This folder contains the images that display the result of Grad-CAM algorithm

##### src
* This folder contains the .py files for model architecture, training procedure, testing procedure, and Grad-CAM algorithm.

##### run.py
* This `run.py` file will the specified target.

##### submission.json
* `submission.json` contains the general structure of this repo.

### How to run this repo with explanation:
*  Please visit the `EDA.ipynb` inside the `notebook folder` to understand the COCO dataset. if github fails to display the image inside the notebook, please refer to `EDA.pdf`. Also, feel free to execute `EDA.py` to generate other set of images for visualization.

To run this repo, run the following lines in a terminal

```
launch-scipy-ml-gpu.sh
git clone https://github.com/gatran/DSC180B-Face-Mask-Detection
cd DSC180B
python run.py test
```

##### Contributions
* Gavin Tran: train the model and generate output
* Che-Wei Lin: implement gradcam and generate output
* Athena Liu: create a report based on those outputs
