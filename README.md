# Temporal_Classification_with_Explainer
In this repository, I'll explore the Sequential Data modelling using deep learing and their explainability. This repo is an work on progress and requires a lot of cleaning, fine tuning etc. 

Right now, I'm running my experiments on a language classification dataset. The purpose of using a language dataset and model is that, it is easier to understand and visualize. Later, I'll focus on the explainability task of other sequential data types.

**Please see the visualizer notebook for details**

## How to run:
1. Run `conda env create -f environment.yml` to create a conda environment.
2. Download the `Natural Language Processing with Disaster Tweets` dataset from [here](https://www.kaggle.com/competitions/nlp-getting-started/data). 
3. Activate your conda environment and run `train.py`. It will train a model for some epochs and save the latest model as `model.pth` file
4. Open the `visualizer.ipynb` notebook and run the cells.

## TODO List
- Implementation of the following papers:
  - *Transformer Interpretability Beyond Attention Visualization*
  - *XAI for Transformers: Better Explanations through Conservative Propagation*
- Explore sequential data
- Integrate Pytorch `Captum` for explainability
- Data augmentation
- Random Seed
- Parameter Tuning
- Logging results using `Tensorboard` or `Wandb`