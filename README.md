# Temporal_Classification_with_Explainer
In this repository, I'll explore the Sequential Data modelling using deep learing and their explainability. This repo is an work on progress and requires a lot of cleaning, fine tuning etc. 

The current implementation has a basic transformer archietecture now. I am using a very small timeseries dataset now. The performance is not sartisfactory at the moment and the model tends to overfit a lot.  

**You can see my result visualization [here](https://wandb.ai/tahsin/Accelerometer%20Project/).**

## How to run:
1. Run `conda env create -f environment.yml` to create a conda environment.
2. Add your data filepath to the `config.py` file. 
3. Activate your conda environment and run `train.py`. It will train a model for some epochs and save the latest model as `model.pt` file
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