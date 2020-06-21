# Inverse Rendering for Complex Indoor Scenes: Shape, Spatially-Varying Lighting and SVBRDF From a Single Image
Zhengqin Li, Mohammad Shafiei, Ravi Ramamoorthi, Kalyan Sunkavalli, Manmohan Chandraker

## New Datasets 
This is the official code release of paper [Inverse Rendering for Complex Indoor Scenes: Shape, Spatially-Varying Lighting and SVBRDF From a Single Image](https://drive.google.com/file/d/17K3RrWQ48gQynOhZHq1g5sQgjLjoMiPk/view). The original models are trained on an extension of SUNCG dataset. Due to the copy right issue, we are not able to release the those models. Instead, we rebuilt a new high-quality synthetic indoor scene dataset and trained our models on it. We will open source the new dataset in the near future. The geometry configurations of the new dataset are based on the ScanNet [1], which is a large-scale repository of 3D scans of real indoor scenes. Some example images can be found below. A video can be found from the [link](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/dataset.mp4)
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/dataset.png)
Insverse rendering results of the models trained on the new datasets are shown below. 
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/inverseRendering.png)
Scene editing applications results on real images are shown below, including results on object insertion and material editing.
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/objectInsertion.png)
Models trained on the new dataset achieve comparable performances compared with our prior models. Quantitaive comparisons are listed below.

