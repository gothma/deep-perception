# Deep Perception


*Environment perception using deep learning techniques*

## Repository structure

    |_ data
    |  |_ images
    |  |  |_ training
    |  |     |_ image_2 // KITTI train images are here
    |  |  |_ testing
    |  |     |_ image_2 // KITTI Test images are here
    |  |__labels
          |_ labels_2 // KITTI Training labels are here
    |_ lib // External code is here

## Data structure

    #Values    Name      Description
    ----------------------------------------------------------------------------
       1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                         'Misc' or 'DontCare'
       1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image boundaries
       1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
       1    alpha        Observation angle of object, ranging [-pi..pi]
       4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
       3    dimensions   3D object dimensions: height, width, length (in meters)
       3    location     3D object location x,y,z in camera coordinates (in meters)
       1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
       1    score        Only for results: Float, indicating confidence in
                         detection, needed for p/r curves, higher is better.

## Todo
- [x] Download dataset 
- [x] Load the data into our current model
- [] Listen to YT lectures and read into papers to get a more profund idea of the layers of a convolutional network
- [] Define first model for dataset
- [] Sliding window with cascade classifiers
- [] Sliding window should work with different patch sizes and strides

## Ideas

- Downscale images before training

## Questions
- How do conv nets correctly handle input images of different sizes? What is the best way for downsampled images? Can we feed conv nets different sizes
- More Ressources what the single layers in a conv network are actually doing.


## Links
* [ML Homepage](http://ml.informatik.uni-freiburg.de/teaching/ws1314/dl)
* [Task Instructions](http://ml.informatik.uni-freiburg.de/_media/teaching/ws1314/dl/10-working_phase_3.pdf)
* [Data Set](http://www.cvlibs.net/datasets/kitti/eval_object.php)


