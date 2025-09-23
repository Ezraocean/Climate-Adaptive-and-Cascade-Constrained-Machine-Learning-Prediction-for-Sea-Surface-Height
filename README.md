Climate-Adaptive and Cascade-Constrained Machine Learning Prediction for Sea Surface Height
This repository implements two machine learning models for predicting sea surface height (SSH) under greenhouse warming conditions. The project focuses on evaluating climate adaptability and introducing physics-informed constraints through kinetic energy cascade mechanisms.

Project Overview
The research develops two ConvLSTM-based models to predict SSH at the Kuroshio Extension region. The Dual-Attention-ConvLSTM Model (DAM) integrates spatial and temporal attention mechanisms with a novel trend-magnitude loss function. Building upon DAM, the Cascade-Constrained-ConvLSTM Model (CCM) incorporates kinetic energy cascade constraints as physics-informed regularization. Both models are trained on historical climate data (1981-2010) and tested on present (2012-2019) and future greenhouse warming scenarios (2092-2099) to evaluate their climate adaptability.

Repository Structure
The main training scripts are and for the respective models, while and handle model evaluation and prediction generation. The core model implementations are contained in imported modules, with results and predictions saved to designated output directories.
Data Requirements
The models require SSH data in MATLAB format for training, present climate testing, and future climate scenarios. Longitude and latitude grid information is also needed for geostrophic velocity calculations. Users need to update the file paths in the scripts according to their data storage location. The daily SSH data used in this study are available from the World Data Center for Climate at DKRZ (https://doi.org/10.26050/WDCC/C6sCMAWAWM and https://doi.org/10.26050/WDCC/C6sSPAWAWM).

Installation and Usage
Install the required dependencies including PyTorch, NumPy, scikit-learn, matplotlib, h5py, scipy, tqdm, and seaborn. For training, simply run the respective training scripts with optional GPU specification. The models use 21 days of input data to predict 7 days ahead by default. Testing scripts generate comprehensive evaluation metrics, prediction visualizations, and performance analysis across multiple statistical and physical measures.

Key Features
The DAM model employs dual attention mechanisms to capture spatial SSH variability patterns and identify critical temporal dependencies. Its trend-magnitude loss function combines traditional MSE with temporal evolution patterns and magnitude sensitivity weighting. The CCM model extends this framework by incorporating spectral kinetic energy flux constraints at multiple scales, ensuring physically consistent cross-scale energy transfers during the learning process. This represents the first application of kinetic energy cascade as a constraint in machine learning-based ocean prediction.

