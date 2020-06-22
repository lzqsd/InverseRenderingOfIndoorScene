# Inverse Rendering for Complex Indoor Scenes: Shape, Spatially-Varying Lighting and SVBRDF From a Single Image
Zhengqin Li, Mohammad Shafiei, Ravi Ramamoorthi, Kalyan Sunkavalli, Manmohan Chandraker

## New Datasets 
This is the official code release of paper [Inverse Rendering for Complex Indoor Scenes: Shape, Spatially-Varying Lighting and SVBRDF From a Single Image](https://drive.google.com/file/d/17K3RrWQ48gQynOhZHq1g5sQgjLjoMiPk/view). The original models are trained on an extension of SUNCG dataset. Due to the copy right issue, we are not able to release the those models. Instead, we rebuilt a new high-quality synthetic indoor scene dataset and trained our models on it. We will open source the new dataset in the near future. The geometry configurations of the new dataset are based on the ScanNet [1], which is a large-scale repository of 3D scans of real indoor scenes. Some example images can be found below. A video can be found from the [link](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/dataset.mp4)
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/dataset.png)
Insverse rendering results of the models trained on the new datasets are shown below. 
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/inverseRendering.png)
Scene editing applications results on real images are shown below, including results on object insertion and material editing.
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/objectInsertion.png)
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/materialEditing.png)
Models trained on the new dataset achieve comparable performances compared with our prior models. Quantitaive comparisons are listed below. [Li20] represents our prior models trained on SUNCG-related dataset. 
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/quantitative.png)

## Download the trained models
The trained models can be downloaded from the [link](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/models.zip). To test the models, please copy the models to the same directory as the code and run the commands as shown below. 

## Train and test on the synthetic dataset
To train the full models on the synthetic dataset, please run the following commands.
* `python trainBRDF.py --cascadeLevel 0 --dataRoot DATA`: 
* ``
* ``
*
To test the full models, pleas

## Train and test on IIW dataset

## Train and test on NYU dataset

## Train and test on Garon19 [2] dataset 

## Differences from the original 
The current implementation has 2 major differences from the original CVPR20 implementation.
* In the new models, we do not use spherical Gaussian parameters generated from optimization for supervision. That is mainly because the optimization proceess is time consuming and we have not finished that process yet. We will update the code once it is done.
* The resolution of the second cascade is changed from 480x640 to 240x320. We find that the networks can generate smoother results with smaller resolution.

## Reference 
[1] Dai, A., Chang, A. X., Savva, M., Halber, M., Funkhouser, T., & Nießner, M. (2017). Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 5828-5839).

[2] Garon, M., Sunkavalli, K., Hadap, S., Carr, N., & Lalonde, J. F. (2019). Fast spatially-varying indoor lighting estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 6908-6917).
