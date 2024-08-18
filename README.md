# EfficientNet
EfficientNet implementation from scratch for classifying images from the Oxford-IIIT Pet Dataset

## Project Structure
- `efficientnet/`: Contains the model and utility functions.
- `data/`: Data transformations.
- `train.py`: Script for training the model.
- `evaluate.py`: Script for evaluating the model.
- `requirements.txt`: List of dependencies.
- `efficientnet_pet_model.pth`: Pre-trained weights provided for your convenience.

## Usage

### 1. Training the Model
To train the model, run:
```
python train.py
```

### 2. Evaluating the Model
First download your desired image and save it in the main directory of project with the name `example.jpg`

Then run the `evaluate.py` script to make predictions using the trained model:
```
python evaluate.py
```

### Dependencies
Install the required dependencies using:
```
pip install -r requirements.txt
```

## Dataset
The dataset used for training is the Oxford-IIIT Pet Dataset.
![Oxford-IIIT Pet Dataset Statistics](https://www.robots.ox.ac.uk/~vgg/data/pets/breed_count.jpg)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- The EfficientNet model architecture is inspired by the original [EfficientNet paper](https://arxiv.org/abs/1905.11946).
- To implement the architecture, [this post on Medium](https://medium.com/@aniketthomas27/efficientnet-implementation-from-scratch-in-pytorch-a-step-by-step-guide-a7bb96f2bdaa) has been used.
