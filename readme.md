# [Speech Understanding] Programming Assignment 3

This README document provides a comprehensive explanation of the provided code

#### Environment requirements

Required Packages
* python 3.7
* fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
* torch==1.8.1+cu111
* torchvision==0.9.1+cu111 
* torchaudio==0.8.1
* gradio
* librosa

#### Important

* The model checkpoints should be present in a folder named `check` in the root directory.
* The datasets should be present as follows:
    - The FOR dataset should be present in a folder named `for-2seconds` in the root directory.
    - The Custom dataset can be simply downloaded in the root directory

### Instructions to run

1. Run the `evaluate.py` file to evaluate the model on the required dataset. Command to run: `python eval.py --path <path to model> --dataset <path to dataset>`
2. Run the `finetune.py` file to finetune the model on the required dataset. Command to run: `python finetune.py --path <path to model> --dataset <path to dataset>`
3. Run the `app.py` file to run the web application. Command to run: `python app.py --path <path to model>`
