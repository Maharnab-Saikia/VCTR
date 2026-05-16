# Official PyTorch implementation of VCTR: A Transformer-Based Model for Non-parallel Voice Conversion.

## Overview
This repository provides the official implementation of the VCTR model for non-parallel voice conversion, as introduced in our paper:

### [Paper](https://arxiv.org/abs/2510.12964)

VCTR is the result of an independent research project focused on non-parallel voice conversion for conversational AI systems.

<p align="center">
  <img src='figs/VCTR.png' width=800>
</p>

This implementation is based on [CVC](https://github.com/Tinglok/CVC). Special thanks to Tinglok for sharing the code.

## Prerequisites
- **Operating System:** Windows, Linux or macOS
- **Python Version:** Python 3.x
- **Hardware:** CPU or NVIDIA GPU with CUDA & CuDNN

## Installation
### Clone the Repository
```bash
git clone https://github.com/Maharnab-Saikia/VCTR
cd VCTR
```

### Install Dependencies
- **For pip users:**
```bash
pip install -r requirements.txt
```
- **For Conda users:**
```bash
conda env create -f environment.yaml
```

### Download Pre-trained Vocoder
Download the pre-trained [Parallel WaveGAN](https://drive.google.com/drive/folders/1qoocM-VQZpjbv5B-zVJpdraazGcPL0So?usp=drive_open) vocoder and place it in the `./checkpoints/vocoder/` directory.

---

## Training and Testing

### Dataset Preparation
Download the **VCTK** dataset:
```bash
wget http://datashare.is.ed.ac.uk/download/DS_10283_2651.zip
unzip DS_10283_2651.zip
unzip VCTK-Corpus.zip
mkdir dataset
cp -r ./VCTK-Corpus/wav48/p* ./datasets/trainA
cp -r ./VCTK-Corpus/wav48/p* ./datasets/trainB
```

### Train the VCTR Model
Run the following command to train the model:
```bash
python train.py --dataroot ./datasets --name VCTR
```
Checkpoints will be saved in `./checkpoints/VCTR/`.

### Test the VCTR Model
```bash
python test.py --dataroot ./datasets --validation_A_dir ./datasets/trainA --output_A_dir ./checkpoints/VCTR/converted_sound
```

Converted audio samples will be saved in `./checkpoints/VCTR/converted_sound/`.

---

## Citation
If you use this code for your research, please cite our [paper](https://arxiv.org/abs/2510.12964).

```
@misc{saikia2025vctrtransformerbasedmodelnonparallel,
      title={VCTR: A Transformer-Based Model for Non-parallel Voice Conversion}, 
      author={Maharnab Saikia},
      year={2025},
      eprint={2510.12964},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2510.12964}, 
}
```
