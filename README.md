<div id="top"></div>

![GitHub](https://img.shields.io/github/license/yakupbilen/drl-rubiks-cube?color=%23009900)
![GitHub last commit](https://img.shields.io/github/last-commit/yakupbilen/drl-rubiks-cube?color=%23009900)
![GitHub repo size](https://img.shields.io/github/repo-size/yakupbilen/drl-rubiks-cube?color=%23009900)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/bilenyakup/)

# Solving Rubik's Cube with Deep Reinforcement Learning and A*.
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#overview">Overview</a>
    </li>
    <li>
      <a href="#motivation">Motivation</a>
    </li>
    <li><a href="#install">Installation</a></li>
    <li><a href="#docker">Running Via Docker</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#techs">Technologies Used</a></li>
    <li>
      <a href="#results">Results</a>
    </li>
    <li><a href="#aspect">Technical Aspect</a></li>
    <li><a href="#bug-feature">Bug / Feature Request</a></li>
    <li><a href="#to-do">To Do</a></li>
    <li><a href="#license">Lıcense</a></li>
    <li><a href="#reference">References</a></li>
  </ol>
</details>

## Overview
This project aimed to solve Rubik's Cube succesfully, using only AI methods.

For any question, please don't hesitate to contact me (bilenyakup97@gmail.com)

<div id="overview"></div>

<p align="center">
  <img alt="GUI App" src="https://drive.google.com/uc?export=view&id=1-5sON1qsWSfq4cHl2gyZmXhNzTEWe3y5">
</p>

<p align="right"><a href="#top">Back to top</a></p>

## Motivation

<div id="motivation"></div>

The Rubik's Cube has ***43,252,003,274,489,856,000*** different state spaces. Only one of them is the solved state. So this is an NP problem.

The motivation of this project is to solve a puzzle with such a large state space with the help of Neural Networks.

<p align="right"><a href="#top">Back to top</a></p>

## Installation

<div id="install"></div>

This project is written in Python 3.9. 


To download this repository, run this command in git bash.
```bash 
  git clone https://github.com/yakupbilen/drl-rubiks-cube
```


To install the required packages and libraries, run this command in the project directory after cloning the repository.

```bash 
  pip install -r requirements.txt 
```

<p align="right"><a href="#top">Back to top</a></p>

## Running Via Docker
<div id="docker"></div>

```bash 
  docker build -t drl-rubiks-cube .
```
#### *Windows*
Install  <a href="https://sourceforge.net/projects/vcxsrv/" target="_blank">VcXsrv Windows X Server</a>. After you install X Server, launch X Server. 
In Extra Settings set checked 'Disable access control'.

```bash 
  docker run --rm -it -e DISPLAY=YourIpAdress:0.0 drl-rubiks-cube
```
#### *Lınux*
```bash 
  docker run --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix drl-rubiks-cube
```
#### *Mac*
Install <a href="https://www.xquartz.org/" target="_blank">xQuartz</a>. Launch xQuartz, preferences->security set checked "Allow connections from network clients".
```bash 
  xhost + YourIpAddress
```
```bash 
  docker run --rm -e DISPLAY=YourIpAddress:0 -v /tmp/.X11-unix:/tmp/.X11-unix drl-rubiks-cube
```

<p align="right"><a href="#top">Back to top</a></p>

## Usage

<div id="usage"></div>

Before training the model, you can make changes in the config(.ini) file that you will send to the program 
as a parameter or you can add new model architecture to models file.

You must to run all of the below codes in this section from project directory.


To train the neural network, you need to run the "run_train.py" file. ***To avoid long hours running***: 
after each run, trained neural network is saved to models/modelname/tmp/model_last.dat.
In each run, program read the trained neural network model and continue to train the model. 
***This process provide us to train the neural network multiple times with short times.***


To train neural network model
```bash 
  python run_train.py -p "path/train_config.ini"
```


To analyze checkpoints
```bash 
  python run_checkpoints_evaluate.py -p "path/analysis_config.ini"
```

To find best parameter for A*
```bash 
  python run_search_tune.py -p "path/analysis_config.ini"
```

To solve Rubik's cube.
```bash 
  python run_solve.py -p "path/solve_config.ini"
```

<p align="right"><a href="#top">Back to top</a></p>



## Technologies Used

<div id="techs"></div>

PyTorch             |  NumPy
:-------------------------:|:-------------------------:
[![torch](https://miro.medium.com/max/160/1*IMGOKBIN8qkOBt5CH55NSw.png)](https://pytorch.org)  |  [![numpy](https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/160px-NumPy_logo_2020.svg.png)](https://numpy.org)

<p align="right"><a href="#top">Back to top</a></p>

## Technical Aspect

<div id="aspect"></div>

This project is divided into three part:

-  Rubik's Cube
    - Defining Rubik's Cube with numpy.
    - Defining moves with numpy indexing.
-  Neural Network Model
    - Training Neural Network.
    - Analyzing and Finding best model among the checkpoints.
-  Solve
    - Solving Rubik's Cube with A*

<p align="right"><a href="#top">Back to top</a></p>

## Results

<div id="results"></div>

Neural network trained on Colab with free Tesla GPUs.

Scrambled cubes are solved with i5-8250u.

![15result](https://drive.google.com/uc?export=view&id=1Yyo1GVGCdqA2PkKRu8KT_u5Rh2qbKMsC)


![100result](https://drive.google.com/uc?export=view&id=1ATr6Yktqr04_PMWPtLwOqjBUzgfMppTc)

<p align="right"><a href="#top">Back to top</a></p>

## Bug / Feature Request

<div id="bug-feature"></div>

If you find a bug kindly open an issue <a href="https://github.com/yakupbilen/drl-rubiks-cube/issues/new" target="_blank">here</a> by the expected result.

If you'd like to request a new function, feel free to do so by opening an issue <a href="https://github.com/yakupbilen/drl-rubiks-cube/issues/new" target="_blank">here</a>. 

<p align="right"><a href="#top">Back to top</a></p>

## To Do

<div id="to-do"></div>

- Train neural network with more data.
- Try different neural network architecture and parameters for better results.
- Implement 3D rubik's cube input from webcam

<p align="right"><a href="#top">Back to top</a></p>

## License

<div id="license"></div>

Distributed under the MIT License. See LICENSE.txt for more information.

<p align="right"><a href="#top">Back to top</a></p>

## References

<div id="reference"></div>

- *McAleer, S., Agostinelli, F., Shmakov, A., & Baldi, P. (2018). Solving the Rubik's cube without human knowledge.* arXiv preprint arXiv:1805.07470
- *Agostinelli, F., McAleer, S., Shmakov, A., & Baldi, P.(2019). Solving The Rubik’s Cube With Deep Reinforcement Learning And Search. Nature Machine Intelligence Vol 1 August 2019 356-363*