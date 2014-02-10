# Deep Perception
## Ramin & Martin

----

## Models
### Multilabel model
* 8-class 
* DontCare not included
* 64, 64, 128

3 stage convolutional network

    64 x 14 x 14 feature map
    64 x  5 x  5 feature map
    dual layer   perceptron

----

### Binary model

* Object / None

Smaller network of same structure

    32 x 14 x 14 feature map
    32 x  5 x  5 feature map
    dual layer   perceptron

----

## Data extraction
### Generally

* Patches at 32 x 32
* Rescaling
   * Insert gray bars <br>
     Preserves ratio
   * Downsampling (Scaling)
* Distortion by mirroring
* Local Spatial Contrastive Normalization

----

### None patches

* Extract square image for each object
* Could contain object but probability for full hit is rather low.
* Global normalization parameters for testing extracted
* Tried extraction in greyscale space
<!-- Explanation graphic here -->

----

## Training

* 18 epoches for multi label classifier
* 3 epoches for binary label classifier <br>
  More time consuming
<!-- Images here -->

----

## Testing

* Minimal patch size 64x64
* Scale factor of 1.3x
* Stride factor of 0.1x
* 11176 samples per image

### Pipeline

* Label samples that pass binary classifier and threshold
* Non maxima suppression at the very end
  * Overlap of 0.2 suffices

----

## Results

![](20.jpg)
![](1344.jpg)
![9](9.jpg)









