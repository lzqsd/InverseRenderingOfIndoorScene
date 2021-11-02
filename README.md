# Inverse Rendering for Complex Indoor Scenes: <br> Shape, Spatially-Varying Lighting and SVBRDF <br> From a Single Image <br> ([Project page](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/))
Zhengqin Li, Mohammad Shafiei, Ravi Ramamoorthi, Kalyan Sunkavalli, Manmohan Chandraker

## Useful links:
* Project page: http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/
* Trained models: https://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/models.zip
* Object insertion: https://github.com/lzqsd/VirtualObjectInsertion
* Tileable texture synthesis: https://github.com/lzqsd/TileableTextureSynthesis
* Spherical gaussian optimization: https://github.com/lzqsd/SphericalGaussianOptimization
* Dataset: https://ucsd-openrooms.github.io/ 

## Results on our new dataset 
This is the official code release of paper [Inverse Rendering for Complex Indoor Scenes: Shape, Spatially-Varying Lighting and SVBRDF From a Single Image](https://drive.google.com/file/d/18zG1kzVpL9XsEVBK95hbpnB-FMlChRXP/view). The original models were trained by extending the SUNCG dataset with an SVBRDF-mapping. Since SUNCG is not available now due to copyright issues, we are not able to release the original models. Instead, we rebuilt a new high-quality synthetic indoor scene dataset and trained our models on it. We will release the new dataset in the near future. The geometry configurations of the new dataset are based on  ScanNet [1], which is a large-scale repository of 3D scans of real indoor scenes. Some example images can be found below. A video is at this [link](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/dataset.mp4)
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/dataset.png)
Insverse rendering results of the models trained on the new datasets are shown below. 
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/inverseRendering.png)
Scene editing applications results on real images are shown below, including results on object insertion and material editing.
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/objectInsertion.png)
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/materialEditing.png)
Models trained on the new dataset achieve comparable performances compared with our previous models. Quantitaive comparisons are listed below, where [Li20] represents our previous models trained on the extended SUNCG dataset. 
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/quantitative.png)

## Download the trained models
The trained models can be downloaded from the [link](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/models.zip). To test the models, please copy the models to the same directory as the code and run the commands as shown below. 

## Train and test on the synthetic dataset
To train the full models on the synthetic dataset, please run the commands
* `python trainBRDF.py --cuda --cascadeLevel 0 --dataRoot DATA`: Train the first cascade of MGNet. 
* `python trainLight.py --cuda --cascadeLevel 0 --dataRoot DATA`: Train the first cascade of LightNet.
* `python trainBRDFBilateral.py --cuda --cascadeLevel 0 --dataRoot DATA`: Train the bilateral solvers.
* `python outputBRDFLight.py --cuda --dataRoot DATA`: Output the intermediate predictions, which will be used to train the second cascade. 
* `python trainBRDF.py --cuda --cascadeLevel 1 --dataRoot DATA`: Train the first cascade of MGNet. 
* `python trainLight.py --cuda --cascadeLevel 1 --dataRoot DATA`: Train the first cascade of LightNet.
* `python trainBRDFBilateral.py --cuda --cascadeLevel 1 --dataRoot DATA`: Train the bilateral solvers.

To test the full models on the synthetic dataset, please run the commands
* `python testBRDFBilateral.py --cuda --dataRoot DATA`: Test the BRDF and geometry predictions.
* `python testLight.py --cuda --cascadeLevel 0 --dataRoot DATA`: Test the light predictions of the first cascade.
* `python testLight.py --cuda --cascadeLevel 1 --dataRoot DATA`: Test the light predictions of the first cascade.

## Train and test on IIW dataset for intrinsic decomposition
To train on the IIW dataset, please first train on the synthetic dataset and then run the commands:
* `python trainFineTuneIIW.py --cuda --dataRoot DATA --IIWRoot IIW`: Fine-tune the network on the IIW dataset.

To test the network on the IIW dataset, please run the commands
* `bash runIIW.sh`: Output the predictions for the IIW dataset.
* `python CompareWHDR.py`: Compute the WHDR on the predictions.

Please fixing the data route in `runIIW.sh` and `CompareWHDR.py`. 

## Train and test on NYU dataset for geometry prediction
To train on the BYU dataset, please first train on the synthetic dataset and then run the commands:
* `python trainFineTuneNYU.py --cuda --dataRoot DATA --NYURoot NYU`: Fine-tune the network on the NYU dataset.
* `python trainFineTuneNYU_casacde1.py --cuda --dataRoot DATA --NYURoot NYU`: Fine-tune the network on the NYU dataset.

To test the network on the NYU dataset, please run the commands
* `bash runNYU.sh`: Output the predictions for the NYU dataset.
* `python CompareNormal.py`: Compute the normal error on the predictions.
* `python CompareDepth.py`: Compute the depth error on the predictions.

Please remember fixing the data route in `runNYU.sh`, `CompareNormal.py` and `CompareDepth.py`.

## Train and test on Garon19 [2] dataset for object insertion
There is no fine-tuning for the Garon19 dataset. To test the network, download the images from this [link](https://lvsn.github.io/fastindoorlight/). And then run `bash runReal20.sh`. Please remember fixing the data route in `runReal20.sh`.

All object insertion results and comparisons with prior works can be found from this [link](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/objectInsertion.zip). The code to run object insertion can be found from this [link](https://github.com/lzqsd/VirtualObjectInsertion). 

## Differences from the original paper
The current implementation has 3 major differences from the original CVPR20 implementation.
* In the new models, we do not use spherical Gaussian parameters generated from optimization for supervision. That is mainly because the optimization proceess is time consuming and we have not finished that process yet. We will update the code once it is done. The performance with spherical Gaussian supervision is expected to be better.
* The resolution of the second cascade is changed from 480x640 to 240x320. We find that the networks can generate smoother results with smaller resolution.
* We remove the light source segmentation mask as an input. It does not have a major impact on the final results.

## Reference 
[1] Dai, A., Chang, A. X., Savva, M., Halber, M., Funkhouser, T., & Nießner, M. (2017). Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 5828-5839).

[2] Garon, M., Sunkavalli, K., Hadap, S., Carr, N., & Lalonde, J. F. (2019). Fast spatially-varying indoor lighting estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 6908-6917).
