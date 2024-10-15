# [Pytorch] A disentangled generative model for disease decomposition in chest X-rays via normal image synthesis

This repository implements Disentangled Generative Model(DGM) with MIMIC-CXR dataset.

**See also**: [Paper](https://doi.org/10.1016/j.media.2020.101839), [tensorflow implementation](https://github.com/YeongHyeon/DGM-TF)

## Data preprocessing

We only use PA sided and clearly labeled data.

Run the shell script bellow

```bash
python data_preprocess_all.py \
    --base_path <your/mimic-cxr-jpg/directory/ends/with/mimic-cxr-jpg/2.1.0/files> \
    --pa_base_path <your/pa_filetered_image/directory/ends/with/physionet.org/pa_filter/pa_filtered_images> \
    --csv_file </standard/test-train/split/in/mimic-cxr/mimic-cxr-2.0.0-split.csv> \
    --label_csv_file </your/mimic-cxr/chexpert/label/path/mimic-cxr-2.0.0-chexpert.csv> \
    --output_path </your/data/output/dir>
```

Then, 13 files would be created in your output directory. We will use `labeled_train.txt`, `labeled_validation.txt`, `labeled_test.txt` in our training code.

## Training

Run the shell script bellow

```bash
python train.py \
    --batch_size 64 \
    --d_steps 1 \
    --gpu <0 indexed gpu number> \
    --train_file_list </your/labeled_train.txt/file> \
    --val_file_list </your/labeled_validation.txt/file>
```

You could modify run script depend on your environment.

## Result

Working..