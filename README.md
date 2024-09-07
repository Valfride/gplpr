# GP_LPR Re-implementation

The original model architecture for the Optical Character Reader (OCR) used in this work is based on the [GP_LPR](https://github.com/MMM2024/GP_LPR) model by Liu et al., as presented in their publication titled *"Irregular License Plate Recognition via Global Information Integration."*

# RodoSol-ALPR Dataset

The [RodoSol-ALPR](https://github.com/raysonlaroca/rodosol-alpr-dataset) dataset contains 20,000 images of various vehicles, including cars and motorcycles. These images were captured under different conditions—day and night, across multiple toll stations, and during both clear and rainy weather.

The dataset is divided into four categories, with 5,000 images in each, based on the type of vehicle and license plate (LP) format:

- **Cars with Brazilian LPs**
- **Motorcycles with Brazilian LPs**
- **Cars with Mercosur LPs**
- **Motorcycles with Mercosur LPs**

### License Plate Formats

- **Brazilian LPs**: Comprise three letters followed by four digits (e.g., ABC-1234).
- **Mercosur LPs**: Follow a specific pattern of three letters, one digit, one letter, and two digits (e.g., ABC1D23), as per the standard in the dataset.

### Arrangement of License Plates

- **Cars**: The license plates contain seven characters displayed in a single row.
- **Motorcycles**: The license plates are arranged across two rows, with three characters in the first row and four characters in the second.


# Usage

This section provides instructions on testing the model, training it from scratch, and fine-tuning it on a custom dataset. Follow the steps below to set up and run the model. Additionally, 

## Testing
To test the model, ensure that the [config file](config/testing.yaml) specifies the path to the .pth file, as shown in the example below:

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

To train or fine-tune the model on a custom dataset, create a `.txt` file that lists the paths to the cropped and rectified images. The file should be formatted as shown below:

```txt
path/to/LP_image1.jpg;training
path/to/LP_image2.jpg;validation
path/to/LP_image3.jpg;testing
```
Next, modify the [config file](configs/training.yaml) to specify the license plate *alphabet* and update the *path_split* argument to point to the .txt file:

```yaml
alphabet: "put the desired alphabet here"

train_dataset:
  dataset:
    name: ocr_img
    args:
      path_split: your_custom_dataset_split.txt
      phase: training
      
  wrapper:
    name: Ocr_images_lp
    args:
      alphabet: "put the desired alphabet here"
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
      path_split: your_custom_dataset_split.txt
      phase: validation

  wrapper:
    name: Ocr_images_lp
    args:
      alphabet: "put the desired alphabet here"
      k: 7
      imgW: 96
      imgH: 32
      aug: False
      image_aspect_ratio: 3
      background: (127, 127, 127)
      with_lr: False
  batch: 128
```
