# Fine-Grained Visual Classification using Self Assessment Classifier

## Prerequisites

PYTHON 3.7 version

CUDA 11.0 version
```
pip install -r requirements.txt
```

## Datasets

- Download CUB-200-2011 dataset (tfrecords) at [link](https://drive.google.com/drive/folders/1F7ba0efDYimk8j3lFkmv_ctBoZHQ7Zqp?usp=sharing) and extract them into `Bird/Data` folder.
- Download FGVC AIRCRAFT dataset (tfrecords) at [link](https://drive.google.com/drive/folders/1laUaBVqEsXMmhwvH1GtrRo2qJEf8Iho1?usp=sharing) and extract them into `Aircraft/Data` folder.
- Download STANFORD DOGS dataset at [link](http://vision.stanford.edu/aditya86/ImageNetDogs/), then convert them into *tfrecords* format and put into `Dog/Data` folder.

#### Dictionary

- Download data dictionary at [link](https://drive.google.com/drive/folders/13JNruqoPoxCwkC_54NPss1S_UdUf4QbW?usp=sharing) and extract them into `data` folder.

## Training

Please download pretrained backbone of WS_DAN at [link](https://drive.google.com/drive/folders/18hHXM4HkDf2T4eaCjpYuqJRHckGqPJVV?usp=sharing) and extract them into `pre_trained` folder.
- To train our method on CUB-200-2011 dataset, please run:
    ```
    bash train_sample_bird.sh
    ```
- To train our method on FGVC AIRCRAFT dataset, please run:
    ```
    bash train_sample_aircraft.sh
    ```
- To train our method on STANFORD DOGS dataset, please run:
    ```
    bash train_sample_dog.sh
    ```

## Testing

#### Evaluate

- To evaluate our method on CUB-200-2011 dataset, please run:
    ```
    bash eval_sample_bird.sh
    ```
- To evaluate our method on FGVC AIRCRAFT dataset, please run:
    ```
    bash eval_sample_aircraft.sh
    ```
- To evaluate our method on STANFORD DOGS dataset, please run:
    ```
    bash eval_sample_dog.sh
    ```
#### Pretrained model

We provide the pretrained model of SAC integrated in WS_DAN on CUB-200-2011 dataset.
- Download our pretrained weights at [link](https://drive.google.com/drive/folders/1F0Butiq8v0LfNxOSD_UtEa11Mu3UkpY6?usp=sharing) and extract them into `Bird/SAC/TRAIN/Bird` folder.

## Citation

If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:

```
Updating
```

## License

MIT License

