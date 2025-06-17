# [Pytorch] A disentangled generative model for disease decomposition in chest X-rays via normal image synthesis

This repository implements Disentangled Generative Model (DGM) with the MIMIC-CXR dataset.

**See also**: [Paper](https://doi.org/10.1016/j.media.2020.101839), [tensorflow implementation](https://github.com/YeongHyeon/DGM-TF)

## Data preprocessing

We only use PA sided and clearly labeled data.
Run the preprocessing script using `preprocess.sh`:

```bash
bash preprocess.sh <base_path> <metadata_csv> <split_csv> <label_csv> <output_dir> [mimic_csv] [chexbert_ckpt] [device]
```

This will generate multiple text files such as `labeled_train_sd.txt` which are used for training.

## Training

Training is invoked via `train.sh`:

```bash
bash train.sh <train_list> <val_list> [batch_size] [d_steps] [gpu]
```

Adjust the optional arguments as needed for your environment.

## Evaluation

To run inference on saved checkpoints use `eval.sh`:

```bash
bash eval.sh <checkpoint> <file_list> <label_csv> [output_dir] [num_samples] [gpu]
```

## Result

Work in progress.
