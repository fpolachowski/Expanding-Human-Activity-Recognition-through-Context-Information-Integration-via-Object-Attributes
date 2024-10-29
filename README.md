# Expanding Human Activity Recognition through Context Information Integration via Object Attributes

This repository contains the Contrastive Human Activity Recognition Model (CHAR). A model for predicting human activities in video data.
The model is setup to be trained on the EPIC-Kitchen dataset.

Pre-trained models will be available soon.

## Setup

1. Download the [EPIC-Kitchen Dataset](https://epic-kitchens.github.io/2020-55.html)
2. Edit the Dockerfile and add the source path of the dataset in the volumes
3. Download the Annotationfile [Link]()
4. Run the packaging script for creating loadable safetensors
5. Setup your Wandb credentials in the Dockerfile
6. Run ```docker compose up -d```
7. Enjoy training
