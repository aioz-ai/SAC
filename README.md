# Fine-Grained Visual Classification using Self Assessment Classifier

## Prerequisites

PYTHON 3.7 version

CUDA 11.0 version
```
pip install -r requirements.txt
```

## Datasets

- Download CUB-200-2011 dataset (tfrecords) at [link](https://vision.aioz.io/f/da1b0ade1c4e4acfbd08/?dl=1) and extract them into `Bird/Data` folder.
- Download FGVC AIRCRAFT dataset (tfrecords) at [link](https://vision.aioz.io/f/cd9a1940688c4795bbdc/?dl=1) and extract them into `Aircraft/Data` folder.
- Download STANFORD DOGS dataset at [link](http://vision.stanford.edu/aditya86/ImageNetDogs/), then convert them into *tfrecords* format and put into `Dog/Data` folder.

#### Dictionary

- Download data dictionary at [link](https://vision.aioz.io/f/5e39ee074cdd446ca9b2/?dl=1) and extract them into `data` folder.

## Training

Please download pretrained backbone of WS_DAN at [link](https://vision.aioz.io/f/be3af5363b9a425cbc7f/?dl=1) and extract them into `pre_trained` folder.
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
- Download our pretrained weights at [link](https://vision.aioz.io/f/97aa30aca9a74ca58bd8/?dl=1) and extract them into `Bird/SAC/TRAIN/Bird` folder.

## Citation

If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:

```
Updating
```

## License

MIT License

