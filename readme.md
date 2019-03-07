# Awesome Human Motion [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> 🏃‍♀️ A curated list about human motion capture, analysis and synthesis.


## Contents

+ [Introduction](#introduction)
+ [Human Models](#human-models)
+ [Datasets](#datasets)
+ [Data Processing](#data-processing)
+ [Pose Estimation](#pose-estimation)
+ [Motion Analysis](#motion-analysis)
+ [Motion Synthesis](#motion-synthesis)
+ [Research Projects](#research-projects)
+ [Commercial Projects](#commercial-projects)
+ [Journals](#journals)
+ [Conferences](#conferences)


## Introduction

+ [Human Pose and Motion](https://github.com/daitomanabe/Human-Pose-and-Motion) - A gentle introduction.


## Human Models

+ [SMPL](http://smpl.is.tue.mpg.de/) - SMPL is a realistic 3D model of the human body that is based on skinning and blend shapes and is learned from thousands of 3D body scans.
+ [MakeHuman](http://www.makehumancommunity.org) - MakeHuman is an open source (AGPL3) tool designed to simplify the creation of virtual humans using a Graphical User Interface, also commonly referred to as a GUI.


## Datasets

+ Sensors and Data Types - 📷(image), 🎥(video), 🎤(audio), 🤾‍♀️(Motion Capture), ⌚️(IMU or wearables), 🤹‍♀️(Kinect or similar)
+ Sex - 👨‍🦰(male), 👩(female)
+ Age - 👧(young), 👵(eldery)

+ [Human 3.6M](http://vision.imar.ro/human3.6m/description.php) - Large Scale Datasets and Predictive Methodsfor 3D Human Sensing in Natural Environments
+ [SURREAL](https://github.com/gulvarol/surreal) - Learning from Synthetic Humans, CVPR 2017
+ [CMU](http://mocap.cs.cmu.edu/) - Carnegie Mellon University Motion Capture Database
+ [Berkley MHAD](http://tele-immersion.citris-uc.org/berkeley_mhad#hard) - [📷🎥🎤🤾‍♀️⌚️🤹‍♀️][👨‍🦰👩][👧👵] - The Berkeley Multimodal Human Action Database (MHAD) contains 11 actions performed by 7 male and 5 female subjects in the range 23-30 years of age except for one elderly subject.
+ [COCO](http://cocodataset.org) - [📷][👨‍🦰👩][👧👵] - COCO is a large-scale object detection, segmentation, and captioning dataset.


## Data Processing

### Recording
+ [Unity Humanoid Mocap CSV](https://github.com/mariusrubo/Unity-Humanoid-Mocap-CSV) - Use .csv files to record, play and evaluate motion capture data.
+ [KinectMotionCapture](https://github.com/apalkowski/KinectMotionCapture) - A simple software for capturing human body movements using the Kinect camera.

### Data Conversion
+ [video to bvh](https://github.com/Dene33/video_to_bvh) - Convert human motion from video to .bvh.
+ [MotionCapturePy](https://github.com/taiyoshoe/MotionCapturePy) - Converts motion capture data from ASF and AMC files to Cartesian numpy arrays in python. Also plots a moving human frame using matplotlib.

### Misc
+ [Motion Annotation Tool](https://github.com/matthiasplappert/motion-annotation-tool) - Crowd-sourced Annotation of Human Motion.


## Pose Estimation

### Lectures
+ [Human Pose Estimation 101](https://github.com/cbsudux/Human-Pose-Estimation-101) - Basics of 2D and 3D Human Pose Estimation.
+ [Object Keypoint Similarity](http://cocodataset.org/#keypoints-eval) - This page describes the keypoint evaluation metrics used by COCO.

### Papers
+ [Papers with Code](https://paperswithcode.com/task/pose-estimation) - A collection of papers addressing several tasks of pose estimation with code available.
+ [Human Pose Estimation Papers](https://github.com/Bob130/Human-Pose-Estimation-Papers) - A collection of papers addressing 2D and 3D human pose estimation.

### Implementations
+ [3Dpose_ssl](https://github.com/chanyn/3Dpose_ssl) - 3D Human Pose Machines with Self-supervised Learning.
+ [3dpose_gan](https://github.com/DwangoMediaVillage/3dpose_gan) - The authors' implementation of Unsupervised Adversarial Learning of 3D Human Pose from 2D Joint Locations.
+ [3d_pose_baseline_pytorch](https://github.com/weigq/3d_pose_baseline_pytorch) - A simple baseline for 3d human pose estimation in PyTorch.
+ [3d-pose-estimation](https://github.com/latte0/3d-pose-estimation) - VNect: Real-time 3D Human Pose Estimation with a Single RGB Camera.
+ [3D-HourGlass-Network](https://github.com/Naman-ntc/3D-HourGlass-Network) - 3D HourGlass Networks for Human Pose Estimation Through Videos.
+ [adversarially_parameterized_optimization](https://github.com/jackd/adversarially_parameterized_optimization) - GAN-based 3D human pose estimation.
+ [DensePose](https://github.com/facebookresearch/DensePose) - A real-time approach for mapping all human pixels of 2D RGB images to a 3D surface-based model of the body
+ [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) - Efficient 3D human pose estimation in video using 2D keypoint trajectorie.
+ [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline) - A simple baseline for 3d human pose estimation in tensorflow. Presented at ICCV 17.
+ [Human Shape and Pose](https://github.com/akanazawa/hmr) -  End-to-end Recovery of Human Shape and Pose - CVPR 2018 


## Motion Analysis

### Implementations
+ [GaitAnalysisToolKit](https://github.com/csu-hmc/GaitAnalysisToolKit) - Tools for the Cleveland State Human Motion and Control Lab.
+ [motion classification](https://github.com/matthiasplappert/motion-classification) - The code written during my Bachelor Thesis "Classification of Human Whole-Body Motion using Hidden Markov Models".
+ [Human-detection-system-with-raspberry-Pi](https://github.com/OmalPerera/Human-detection-system-with-raspberry-Pi) - A motion detection system with RaspberryPi, OpenCV, Python.
+ [humanMotionClassification](https://github.com/ltecot/humanMotionClassification) - Experiments in classifying human actions using the UCF action databased.
+ [sensormotion](https://github.com/sho-87/sensormotion) - Python package for analyzing sensor-collected human motion data (e.g. physical activity levels, gait dynamics).
+ [Posture and Fall Detection System Using 3D Motion Sensors](https://github.com/Health-Devices-Research-Group/Posture-and-Fall-Detection-System-Using-3D-Motion-Sensors) - This work presents a supervised learning approach for training a posture detection classifier, and implementing a fall detection system using the posture classification results as inputs with a Microsoft Kinect v2 sensor.
+ [HumanMotionVisualiser](https://github.com/BayesTech/HumanMotionVisualiser) - This project is for visualising human motion data captured from Kinect V2 for further data analysis.
+ [human motion analysis](https://github.com/dfsdfb/human-motion-analysis)
+ [human motion classification](https://github.com/kubapok/human-motion-classification)
+ [motion visualization](https://github.com/matthiasplappert/motion-visualization) - A simple visualizer for human whole-body motion using three.js


## Motion Synthesis

### Implementations
+ [Auto Conditioned RNN motion](https://github.com/papagina/Auto_Conditioned_RNN_motion) - Implementation of Auto-Conditioned Recurrent Networks for Extended Complex Human Motion Synthesis.
+ [Character Animation](https://github.com/AliJalalifar/Character_Animation) - A Re-implementation of the paper "A Deep Learning Framework for Character Motion Synthesis and Editing".
+ [eccv18_mtvae
](https://github.com/xcyan/eccv18_mtvae) - Tensorflow Implementation of ECCV'18 paper: Multimodal Human Motion Synthesis.
+ [motionSynth](https://github.com/utkarshmall13/motionSynth) - Deep Human Motion Synthesis.
+ [MotionSynthesis2Maya](https://github.com/ArashHosseini/MotionSynthesis2Maya) - create a Maya Pipeline based on Motion Synthesis.
+ [Adversarial Learning for Modeling Human Motion](https://github.com/lucaskingjade/Motion_Synthesis_Adversarial_Learning) - This repository contains the open source code which reproduces the results for the paper: Adversarial learning for modeling human motion.
+ [Human Motion Synthesis](https://github.com/tomatosoldier/Human-Motion-Synthesis) - Human motion synthesis using Unity3D.
+ [GAN motion Prediction](https://github.com/amoghadishesha/GAN-motion-Prediction) - An LSTM based GAN for Human motion synthesis.
+ [Merel MoCap GAIL](https://github.com/ywchao/merel-mocap-gail) - An implementation of "Learning human behaviors from motion capture by adversarial imitation".


## Research Projects

+ [Uni Bonn: Physics-based motion analysis and synthesis](http://cg.cs.uni-bonn.de/en/projects/motion-analysis-and-synthesis/) - Physically-based analysis and synthesis of (human) motions have a number of applications. They can help to enhance the efficiency of medical rehabilitation, to improve the understanding of motions in the realm of sports or to generate realistic animations for movies.
+ [FAU: Biomechanical Simulation for the Reconstruction and Synthesis of Human Motion](https://www.mad.tf.fau.de/research/projects/biomechanical-simulation/) - In this project, we investigate musculoskeletal modeling and simulation to analyze and understand human movement and performance. Our objective is to reconstruct human motion from measurement data for example for medical assessments or to predict human responses for virtual product development.


## Commercial Projects

+ [wrnch.ai](https://wrnch.ai/) - wrnch is a computer vision / deep learning software engineering company based in Montréal, Canada, a renowned hub for AI.
+ [Qinematic](https://www.qinematic.com/) - Qinematic has developed software for 3D markerless motion capture and human movement analysis since 2012.


## Journals

+ [Computers in Biology and Medicine](https://www.journals.elsevier.com/computers-in-biology-and-medicine)
+ [Informatics in Medicine Unlocked](https://www.journals.elsevier.com/informatics-in-medicine-unlocked)
+ [Image and Vision Computing](https://www.journals.elsevier.com/image-and-vision-computing)
+ [Clinical Biomechanics](https://www.journals.elsevier.com/clinical-biomechanics)
+ [Signal Processing: Image Communication](https://www.journals.elsevier.com/signal-processing-image-communication)
+ [Artificial Intelligence in Medicine](https://www.journals.elsevier.com/artificial-intelligence-in-medicine)


## Conferences

+ [ICRA](https://dblp.org/db/conf/icra/index) - International Conference on Robotics and Automation
+ [MICCAI](https://dblp1.uni-trier.de/db/conf/miccai/) - Medical Image Computing and Computer-Assisted Intervention


## Credits

This list benefits massively from the research work of [**Loreen Pogrzeba**](https://www.researchgate.net/profile/Loreen_Pogrzeba).


## Contribute

Contributions welcome! Read the [contribution guidelines](contributing.md) first.


## License

[![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](http://creativecommons.org/publicdomain/zero/1.0)

To the extent possible under law, derikon has waived all copyright and
related or neighboring rights to this work.
