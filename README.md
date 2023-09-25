# Temporal_Classification_with_Explainer
In this repository, I'll explore the Sequential Data modelling using deep learing and their explainability. This repo is an work on progress and requires a lot of cleaning, fine tuning etc. 
  

**You can see my result visualization [here](https://wandb.ai/tahsin/Accelerometer%20Project/).**

## How to run:
1. Run `conda env create -f environment.yml` to create a conda environment.
2. Add your data filepath to the `config.py` file `config_params` dictionary. 
3. Change `model_name` parameter for the dictionary and set it to one of the `mini_transformer`, `resnet`, `squeezenet`, `mobilenet` and `vgg` to train them.
4. If you do not have a `wandb` account, open one at [wandb.ai](wandb.ai). You will be provided with a unique code. 
4. Activate your conda environment and run `train.py`. It will first ask you to log in to your wandb account using the unique code you have. After providing the code, it will start training a model for some epochs, log the results at a given link and save the latest model as `model.pt` file. You can visualize your results at the provided link.  
5. Open the `visualizer.ipynb` notebook and run the cells.

## Implemented Models
- Resnet
- Squeezenet
- Mobilenet
- VGG
- Transformer
- Densenet [Not working]
- LSTM [Not working]


## Directory Graph
```
.
├── AccelerometerDataset.py
├── config.py
├── conf.png
├── environment.yml
├── model_graphs
│   ├── mini_transformer_graph
│   ├── mini_transformer_graph.png
│   ├── mobileresnet1D_graph
│   ├── resnet1D_graph
│   ├── seresnet1D_graph
│   ├── vgg1D_graph
├── models
│   ├── densenet.py
│   ├── lstm.py
│   ├── mini_transformer.py
│   ├── mobilenet.py 
│   ├── resnet.py
│   ├── squeezenet.py
│   └── vgg.py
├── README.md
├── train_module.py
├── train.py
├── utils.py
├── visualizer.ipynb

```

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
