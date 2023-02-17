# Medical Image Subregion Analysis Toolkit (MISA) 

<p align="center">
  <img src="https://user-images.githubusercontent.com/17007301/219617294-a5f38b07-4599-4834-aa7c-96d01299a531.png" width="600" height="300">
</p>

|Build/Test Status|Code Status|Documentation|
| :---------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------: |
| [![](https://travis-ci.org/modelhub-ai/modelhub-engine.svg?branch=master)](https://travis-ci.org/modelhub-ai/modelhub-engine) | [![Coverage Status](https://coveralls.io/repos/github/modelhub-ai/modelhub-engine/badge.svg?branch=master&service=github)](https://coveralls.io/github/modelhub-ai/modelhub-engine?branch=master) | [![Documentation Status](https://readthedocs.org/projects/modelhub/badge/?version=latest)](https://modelhub.readthedocs.io/en/latest/?badge=latest) |

### Welcome to Medical Image Subregion Analysis Toolkit (MISA) package.

MISA is a Python package, which aims for tumor subregion and surrounding microenvironment analysis in cancer domain by medical imaging data, such as CT, PET, MRI and US. Note that, MISA is not only developed for 3D medical data, but also for 2D medical data (e.g., US or single 2D slice). 

MISA is designed and developed by Zhenwei Shi, Zhihe Zhao and other AI/CS scientists from Media Lab. Also, the work is supported and guided by well-known radiologists MD Zaiyi Liu and MD Changhong Liang from the radiolgoy department of Guangdong Provincial People's Hospital.

The workflow of MISA includes five major functionalities:
- Data pre-processing
- Subregion pre-segmentation
- Clustering in population-level
- Intra-tumoral heterogeneity assessment
- Tumor delineation contour perturbation

### Installation

```
pip install MISA
```
### Features

- Medical image data pre-processing, including data load, crop, normalization and so on.
- Automatic generation of multiple regions of interest surrounding tumor, such as peritumor and tumor ring
- Subregion pre-segmentation by image properties
- Quantitative imaging feature (e.g., Radiomics) extraction
- Unsupervised clustering algorithms for untimate medical image subregion partition
- Visualiation of multi-partitions
- Intra-tumoral heterogeneity assessment by anlyzing intra-tumor and surronding microenviourment regions
- Tumor delineation contour perturbation


### Tutorial

#### 1. Package Loading

```
import numpy as np
import matplotlib.pyplot as plt
import os 
from MISA.function import makedirs, extract_main, feature_extract_main, cluster_main, cluster_main_predict
```

#### 2. Parameter Setting
Some local directories are set before processing, including main workding directory, image and mask directory, output directory, and also parameter setting directory for quantitative imaging feature extraction. Note that, MISA uses an open-source software [PyRadiomics](https://pyradiomics.readthedocs.io/en/latest/) to extract radiomic features as default, while the clients are allowed to use any other kinds of features. For radiomics extraction, the clients can download the parameter setting file (.ymal) for simulation from [MISA](https://github.com/zhenweishi/MISA).

```sh
# set local paths
dataset_path = 'dataset'
image_path = dataset_path+'/image'
mask_path = dataset_path+'/mask'
out_path = 'subregion_SLIC_output/SuperVoxel'
yaml_path = 'radiomics_features.yaml'
sv_path = os.path.join(out_path,'supervoxel')
csv_path = os.path.join(out_path,'csv')
concat_path = os.path.join(out_path, 'concat_mask')

# make local paths
makedirs(sv_path)
makedirs(csv_path)
makedirs(concat_path)

<p align="center">
  <img src="https://raw.githubusercontent.com/zhenweishi/MISA/main/Materials/pre-segmentation.png" width="600" height="400">
</p>

```

MISA provides a functionality to automatically generate multiple regions of interest (ROI) surrounding tumor, whcih are able to describe tumor microenvironment, such as peritumor and tumor ring. The clients can change the size of the peritumor or tumor ring area by modifying the kernel_size parameter, and select the subregion processing mode: 'initial (default as original tumor)', 'peritumor', 'tumor ring'.

```sh
mode = 'peritumor' # 'initial','peritumor','tumor_ring'
kernel_size = 3
```
#### 3. Pre-segmentation of subregions

MISA follows a two-step subregion segmentation strategy, that is, pre-segmentation and fine subregion partition. In the pre-segmentation step, it splits the whole ROI in pieces by taking into account image properties itself, of which the number depends on the surface/volume of the ROI. To preserve enough information for feature extraction later, MISA suggests not to split the individual pre-segments too small. Some examples with pre-segmented subregion  maps are shown as follows.

```sh
extract_main(image_path, mask_path, sv_path, out_path, mode, kernel)
```

![pre-segmentation](https://user-images.githubusercontent.com/17007301/219617436-37cf7a37-de46-4574-bcd2-0c070c7dfecd.png)


#### 4. Quantitative imaging feature extraction

MISA uses radiomic feature as the default quantiative imaging feature. In this step, MISA is able to extract radiomics from the small regions aquired above. Also, the clients are allowed to use other kinds of features, such as deep learning and handcraft features.

```sh
feature_extract_main(sv_path, csv_path,yaml_path)
```

#### 5. Generation of subregion partition map

MISA provides a function to cluster the small pre-segmented regions by analyzing the imaging features, which can gether the small regions with similar properties. Some examples with final subregion partition maps are shown as follows.


```sh
cluster_main(image_path, csv_path, sv_path, concat_path, out_path)
```

![Concat_subregionmap](https://user-images.githubusercontent.com/17007301/219617647-edd8599e-2299-47e1-bd4f-21028f1136e6.png)

### License

MISA package is freely available to browse, download, and use for scientific and educational purposes as outlined in the [Creative Commons Attribution 3.0 Unported License](https://creativecommons.org/licenses/by/3.0/).

### Disclaimer

MISA is still under development. Although we have tested and evaluated the workflow under many different situations, it may have errors and bugs unfortunately. Please use it cautiously. If you find any, please contact us and we would fix them ASAP.

### Main Developers
 - [Dr. Zhenwei Shi](https://github.com/zhenweishi) <sup/>1, 2
 - MSc. Zhihe Zhao <sup/>2, 3
 - MSc. Zihan Cao <sup/>2, 4
 - MD. Xiaomei Huang <sup/>2, 5
 - [Dr. Chu Han](https://chuhan89.com) <sup/>1, 2
 - MD. Changhong Liang <sup/>1, 2
 - MD. Zaiyi Liu <sup/>1, 2
 

<sup>1</sup> Department of Radiology, Guangdong Provincial People's Hospital (Guangdong Academy of Medical Sciences), Southern Medical University, China <br/>
<sup>2</sup> Guangdong Provincial Key Laboratory of Artificial Intelligence in Medical Image Analysis and Application, China <br/>
<sup>3</sup> School of Medicine, South China University of Technology, China <br/>
<sup>4</sup> Institute of Computing Science and Technology, Guangzhou University, China <br/>
<sup>5</sup> Department of Medical Imaging, Nanfang Hospital, Southern Medical University, China 

### Contact
We are happy to help you with any questions. Please contact Zhenwei Shi.
Email: shizhenwei@gdph.org.cn

We welcome contributions to MISA.
