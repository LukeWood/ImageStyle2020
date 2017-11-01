# StyleTransfer

Keras 2 + Tensorflow implemenation of [fast-style-transfer](https://github.com/lengstrom/fast-style-transfer)

# Requirements

keras 2  
Tensorflow   
keras_contrib (for InstanceNormalization)  

# Setup

## Training data
Any large image set can be used, we are using MSCOCO:  
- http://msvocds.blob.core.windows.net/coco2014/train2014.zip (~13G)  
- http://msvocds.blob.core.windows.net/coco2014/val2014.zip  (~6.6G)
- http://msvocds.blob.core.windows.net/coco2014/test2014.zip (~6.6G)

Because of the way Keras' ImageDataGenerator looks for images, the images need to be in a subdirectory
of the directory passed as `train-path`, for example if `data` is passed as `train-path`:
```
data
  -train2014
    - 1.jpg
    - 2.jpg
    ...
  -val2014
  -test2014
```

# train.py 
```
python train.py --style wave.jpg --model-output wave.h5
```
### Options
```
  --style STYLE                     style image path
  --model-output MODEL_OUTPUT       path to save the trained model out as a h5 file
  --model-input MODEL_INPUT         path to model to train (if continuing training)
  --test TEST                       test image path, if given will style this image 
                                    after every test-increment and save into test-dir
  --test-dir TEST_DIR               test image save dir
  --test-increment TEST_INCREMENT   number of batches to test after
  --train-path TRAIN_PATH           path to training images folder (default 'data')
  --epochs EPOCHS                   num epochs (default 2)
  --batch-size BATCH_SIZE           batch size (default 4)
  --steps-per-epoch BATCH_SIZE      number of batches of samples per epoch,
                                    should be # of samples / batch size
  --content-weight CONTENT_WEIGHT   content weight (default 15.0)
  --style-weight STYLE_WEIGHT       style weight (default 100.0)
  --tv-weight TV_WEIGHT             total variation regularization weight (default 200.0)
  --conv-filters CONV_FILTERS [CONV_FILTERS ...]
                                    number of filters in conv layers in transform net
  --num-resids NUM_RESIDS
                                    number of residual blocks in transform net
```
# evaluate.py 
```
python evaluate.py --model wave.h5 --input doge.jpg --output doge-wave.jpg
```

### Options
```
  --model MODEL                   model path
  --input INPUT                   input image path
  --output OUTPUT                 output image path
  -p, --pad                       add reflection padding to input image
  -b. --border-size BORDER_SIZE   border size of reflection padding
  --conv-filters CONV_FILTERS [CONV_FILTERS ...]
                                    number of filters in conv layers in transform net
  --num-resids NUM_RESIDS
                                    number of residual blocks in transform net
```
