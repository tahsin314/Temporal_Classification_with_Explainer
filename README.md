# Temporal_Classification_with_Explainer

This repository is dedicated to the classification of ICU patient conditions using data from accelerometer sensors attached to their bodies. The project focuses on sequential data modeling using deep learning techniques and aims to provide insights into the explainability of the classification results. Please note that this repository is a work in progress and may require additional cleaning, fine-tuning, and further development.

You can explore visualizations of the project's results [here](https://wandb.ai/tahsin/Accelerometer%20Project/).

## How to Run the Project

To get started with the project, follow these steps:

1. Create a Conda environment by running the following command:

   ```bash
   conda env create -f environment.yml
   ```

2. Open the `config.py` file and add your data file path to the `config_params` dictionary.

3. In the `config.py` file, set the `model_name` parameter to one of the available models: `mini_transformer`, `resnet`, `squeezenet`, `mobilenet`, or `vgg`.

4. If you don't have a [wandb](https://wandb.ai) (Weights and Biases) account, please create one. You will receive a unique code.

5. Activate your Conda environment and run the training script:

   ```bash
   python train.py
   ```

   The script will prompt you to log in to your wandb account using the unique code you received. After providing the code, it will start training a model for several epochs, log the results at a given link, and save the latest model as `model.pt`. You can visualize your results at the provided link.

6. Open the `visualizer.ipynb` notebook and run the cells to further explore and analyze the results.

## Implemented Models

- Resnet
- Squeezenet
- Mobilenet
- VGG
- Transformer
- Densenet (Currently not working)
- LSTM (Currently not working)

## Directory Structure

```plaintext
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

- Implement research papers:
  - "Transformer Interpretability Beyond Attention Visualization"
  - "XAI for Transformers: Better Explanations through Conservative Propagation"
- Integrate PyTorch `Captum` for explainability.
- Explore data augmentation techniques.
- Implement random seed management for reproducibility.
- Fine-tune model hyperparameters for better performance.

Feel free to contribute to this project or reach out if you have suggestions, bug reports, or feature requests. Your input is valuable to us.

## License

This project is open-source and available under the MIT License.

---

We are continuously working to improve and extend this project. If you have any ideas or recommendations, please share them with us.
