# Spatial Re-parameterization for N:M Sparsity

## Requirements

- python 3.7
- pytorch 1.10.2
- torchvision 0.11.3

## Re-produce our results

### Training
Select a configuration file in configs to reproduce the experiment results reported in the paper. For example, to prune ResNet-50 on ImageNet dataset, run:

- ResNet on ImageNet
```bash
python imagenet.py --config configs/resnet50.yaml --gpus GPU_IDS 
```
Note that the `data`, `N`, `M` in the yaml file should be changed to the data path and N:M pattern.

### Re-parameterization

To perform the re-parameterization that convert the training-time structure into the inference-time structure, run:

```bash
python convert.py --config configs/resnet50.yaml --train_model_link PATH_TO_TRAIN_MODEL
```

Then you will get a converted checkpoint for deployment.

### Testing

Besides, we provide our trained models and experiment logs at [Google Drive](https://drive.google.com/drive/folders/1HlTpJTC2omTRBrTsVPMYrlCpGha-Vd1B?usp=share_link). To test, run:

```bash
python imagenet.py --config configs/resnet50.yaml --evaluate --evaluate_model_link PATH_TO_EVALUATE_MODEL --gpus GPU_IDS 
```

To test the re-parameterized model, add `--deploy` in the above script.

