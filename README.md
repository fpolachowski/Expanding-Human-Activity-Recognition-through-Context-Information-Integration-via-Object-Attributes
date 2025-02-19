# Expanding Human Activity Recognition through Context Information Integration via Object Attributes

This repository contains the Contrastive Human Activity Recognition Model (CHAR). A model for predicting human activities in video data.
The model is setup to be trained on the EPIC-Kitchen dataset.

## Requirements

- Python \>= 3.10
- PyTorch \>= 2.0
- Docker & Docker Compose

## Setup

1. Download the [EPIC-Kitchen Dataset](https://epic-kitchens.github.io/2020-55.html)
2. Edit the Dockerfile and add the source path of the dataset in the volumes
3. Run the packaging script for creating loadable safetensors
4. Setup your Wandb credentials in the Dockerfile
5. Run ```docker compose up -d```
6. Enjoy training
