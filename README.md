# gans v1.0
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/Naereen/)


# Deep Convolutional Generative Adversarial Network (DCGAN*) implemented in PyTorch.

### Applied on different medical datasets for different medical-based approaches (later we will be discussing Segmentation).

Unlike the original paper "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" we generate 128 x 128 Images that looks quite plausible.Check the results above and try it yourself.

![gans](https://github.com/AissamDjahnine/gans/blob/master/imgs/gans_ex.png)

## Motivation

The main objective of this work is twofold : 

* Generate synthetic medical images that model the distribution of the input data 
* [SOON] Use both of the synthetic and real images for training a context-aware CNN that can accurately segment the given images


## Install
In your terminal, run:

```bash
git clone https://github.com/AissamDjahnine/gans
cd gans
```

Then, install all dependencies with PIP using:

```bash
pip install -r requirements.txt
```

# Datasets

## Medical datasets : 

Since the main task behind this project is to generate synthetic medical images , we used these two datasets described below ( links are available as well )

![gans](https://github.com/AissamDjahnine/gans/blob/master/imgs/datasets.png)
### * Absorbance microscopy images of human Retinal Pigment Epithelium (RPE) cells
###
You can use this Dataset consisted of 1032  ( 256 x 256 ) RPE cells images images.You can browse and donwload the data using this link : [RpeCells](https://drive.google.com/drive/folders/1gzqjCvfp6pIpM2aT6UWgD3DkJNdhZ9D-?usp=sharing)

### * Osteosarcoma data from UT Southwestern/UT Dallas for Viable and Necrotic Tumor Assessment

The dataset consists of 1144 images of size 1024 X 1024 at 10X resolution with the following distribution: 536 (47%) non-tumor images, 263 (23%) necrotic tumor images and 345 (30%) viable tumor tiles.

You can browse the data using this link : [Osteosarcoma data from UT Southwestern/UT Dallas](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52756935), to download the data that you can use directly: [Osteosarcoma data](https://drive.google.com/open?id=1mN7Wqbo1FOchtPITS83xkDu-j6C6d74q)

## Other datasets: 

We tried to use the model on other datasets to evaluate its performance on non-medical images.Giben below the dataset we used for this experiment 

### UTKFace dataset

UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization, etc.

In our case , we'll be using our generative model to try to learn data distrubution and generate more look alike images.
You see In Results sbelow the results we obtained after training (500 epochs)

## Usage

### useful notes : 

* You can train the model on big datasets.The script saves your current model state (weights , bias ..) so that you can resume training from this point.the state is saved after 25 epochs (you can adjust the value depending on how big is your dataset) --Available only for the API. 

* Please respect this structure while feeding the model with your images directory :

```bash
./IMAGESFOLDER/Images/.. Examlpe : RPEDATA/Images/.. , Osteosarcoma_data/Images/..
```

### Jupyter Notebook 

Run a jupyter process :

```bash
Jupyter notebook
```

You find a Jupyter notebook version of the project , you can interactively run cells and check outputs for different sections 

![gans](https://github.com/AissamDjahnine/gans/blob/master/imgs/jupyter.jpg)

In your terminal, run:
```bash
mkdir Generated_Images_numpy_notebook
```
To create a directory for the generated imges ( numpy arrays ) version.

### API 

In your terminal, run:

```bash
python GAN_F_128.py --path IMAGEFOLDER --epochs 500 --batch-size 128 --lr 0.0002
```
Note : You may use default values ( see the file : GAN_F_128.py )

#### Example : 

```bash
python GAN_F_128.py --path ./RPEDATA --epochs 500 
```

## Visualization
### Jupyter notebook 

* You can see generated images changing with time ( every 10 batches ), feel free to change the display time.

### Tensorboard 

If you're familiar with Tensorboard , skip this section 

In your terminal, run:

```bash
tensorboard --logdir ./runs
```
* You should be looking to :
![gans](https://github.com/AissamDjahnine/gans/blob/master/imgs/tb_.jpg)

## References 

* "Generative Adversarial Networks." Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. ArXiv 2014.

* "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" Alec Radford, Luke Metz, Soumith Chintala



## Questions :
[![Generic badge](https://img.shields.io/badge/TEST-VERSION-RED.svg)](https://github.com/AissamDjahnine)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://github.com/AissamDjahnine)
[![GitHub followers](https://img.shields.io/github/followers/Naereen.svg?style=social&label=Follow&maxAge=2592000)](https://github.com/AissamDjahnine?tab=followers)

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please feel free to contact if case of issues.
