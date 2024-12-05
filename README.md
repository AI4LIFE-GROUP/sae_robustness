## Introduction
This is the repo for AM207 Course Project: Input-Level Attack Against Sparse Autoencoders With Evolutionary Algorithms.

The APIs related to SAEs are based on the library https://github.com/EleutherAI/sae, with some customized local package changes. The trained SAEs are loaded from https://huggingface.co/EleutherAI/sae-llama-3-8b-32x.

The attack is implemented in ``evolution.py``. Before running it, one should first download the SAEs following instructions from the SAE library released by EleutherAI (same link as above) and change the related local paths for the LLM and SAEs. 

Other code files in this repo are other previously implemented variants of the attack, which are not directly related to this course project. Please ignore them.

