# sign-language-detection

## How to use the repository

Clone the repository
```bash
git clone https://github.com/Bharathjpv/sign-language-detection.git
```

Create an environament and install the requirements
```bash
pip install -r requirements.txt
```

# 1. To collect images using a camera

Run `collect_imgs.py` file

change the number of classes required in using the variable `number_of_classes` in line no 10 as per your requirements.

change the sample size for each classes using the variable `dataset_size` in line no 11 as per your requirements.

```bash
python collect_imgs.py
```
# 2. To create dataset using the images collected.

Run `create_dataset.py` file

```bash
python create_dataset.py
```
This will save your dataset in a pickle file

# 3. Train the model

Here I have chosen Random forest you can use any of your own.

Run `train_classifier.py` file
```bash
python train_classifier.py
```

This will train and save the model into a pickle file.

# 4. Predictions

Predict on real time basis usng camera.

Run `inference_classifier.py` file

```bash
inference_classifier.py
```

This will give you real time predictions.