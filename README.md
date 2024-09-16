# GP_LPR Implementation

The original model architecture for the Optical Character Reader (OCR) used in this work is based on the [GP_LPR](https://github.com/MMM2024/GP_LPR) model by Liu et al., as presented in their publication titled *"Irregular License Plate Recognition via Global Information Integration."*

# RodoSol-ALPR Dataset

The [RodoSol-ALPR](https://github.com/raysonlaroca/rodosol-alpr-dataset) dataset contains 20,000 images of various vehicles, including cars and motorcycles. These images were captured under different conditions—day and night, across multiple toll stations, and during both clear and rainy weather.

The dataset is divided into four categories, with 5,000 images in each, based on the type of vehicle and license plate (LP) format:

- **Cars with Brazilian LPs**
- **Motorcycles with Brazilian LPs**
- **Cars with Mercosur LPs**
- **Motorcycles with Mercosur LPs**

### License Plate Layouts

- **Brazilian LPs**: Comprise three letters followed by four digits (e.g., ABC-1234).
- **Mercosur LPs**: Follow a specific pattern of three letters, one digit, one letter, and two digits (e.g., ABC1D23), as per the standard in the dataset.

### Arrangement of License Plates

- **Cars**: The license plates contain seven characters displayed in a single row.
- **Motorcycles**: The license plates are arranged across two rows, with three characters in the first row and four characters in the second.


# Usage

This section provides instructions on testing the model, training it from scratch, and fine-tuning it on a custom dataset. Follow the steps below to set up and run the model. Additionally, 

## Testing
To test the model, ensure that the config file specifies the path to the .pth file (e.g.,  [config file](config/GP_LPR_RODOSOL_test.yaml)), as shown in the example below:

```yaml
model_ocr:
  name: GPLPR
  load: ./save/testing/model.pth
  args:
    nc: 3
    isSeqModel: True
    head: 2
    inner: 256
    isl2Norm: True
```

Once the configuration is set, execute the following command to start the test:

```
python3 test_ocr.py --config ./config/testing.yaml --save True --tag example
```

## Training from Scratch

To train the model from scratch, update the following variables in the [config file](config/training.yaml):

```yaml
resume: null
```

Optionally, you can add the --tag argument for versioning:
```
python3 train.py --config ./config/training.yaml --save True
```

Optionally, you can add the --tag argument, for versioning:
```
python3 train.py --config ./config/training.yaml --save True --tag example
```

## Training on a Custom Dataset

To train or fine-tune the model on a custom dataset, you need to create a .txt file that lists the image paths along with their corresponding data split (training, validation, or testing). Each line in the file should follow this format:

```txt
path/to/LP_image1.jpg;training
path/to/LP_image2.jpg;validation
path/to/LP_image3.jpg;testing
```
For reference, you can check example files, such as [split_all_pku.txt](split_all_pku.txt) and [split_all_rodosol.txt](split_all_rodosol.txt), which demonstrate this format.


### Modifying the Configuration File for Training/Finetuning

To customize the model for training or fine-tuning on a custom dataset, follow these steps:

1. **Create a custom alphabet**: Update all fields marked with `alphabet: "customAlphabet"` in the config file to reflect your specific alphabet.
2. **Update the dataset split**: Modify the `path_split` argument to point to your custom dataset split file (e.g., `your_custom_split.txt`).

Here’s an example of the modified `training.yaml` file:

```yaml
model:
  name: GPLPR
  OCR_TRAIN: True
  args:
    nc: 3
    alphabet: "customAlphabet"   # Specify your custom alphabet here
    K: 7
    isSeqModel: True
    head: 2
    inner: 256
    isl2Norm: True

func_train: GP_LPR_TRAIN
func_val: GP_LPR_VAL
alphabet: "customAlphabet"         # Apply the custom alphabet

train_dataset:
  dataset:
    name: ocr_img
    args:
      path_split: your_custom_split.txt  # Set the path to your custom split file
      phase: training

  wrapper:
    name: Ocr_images_lp
    args:
      alphabet: "customAlphabet"   # Specify your custom alphabet here
      k: 7
      imgW: 96
      imgH: 32
      aug: True
      image_aspect_ratio: 3
      background: (127, 127, 127)
      with_lr: False
  batch: 128

val_dataset:
  dataset:
    name: ocr_img
    args:
      path_split: your_custom_split.txt  # Set the path to your custom split file
      phase: validation

  wrapper:
    name: Ocr_images_lp
    args:
      alphabet: "customAlphabet"   # Specify your custom alphabet here
      k: 7
      imgW: 96
      imgH: 32
      aug: False
      image_aspect_ratio: 3
      background: (127, 127, 127)
      with_lr: False
  batch: 128

optimizer:
  name: adam
  args:
    lr: 1.e-3
    betas: [0.5, 0.555]

epoch_max: 3000

loss:
  name: CrossEntropyLoss
  args:
    size_average: None
    reduce: None
    reduction: mean

early_stopper:
  patience: 400
  min_delta: 0
  counter: 0

epoch_max: 3000
epoch_save: 100
resume: null   # For fine-tuning, point this to your pre-trained model path
```
### Additional Notes:
- **Fine-tuning**: If you’re fine-tuning the model, modify the `resume:` field to point to your pre-trained model file.
- **Alphabet**: Make sure to specify the correct custom alphabet consistently in all sections.
- **Dataset split**: Ensure the `your_custom_split.txt` file contains the correct paths and data splits (e.g., training, validation, testing).
