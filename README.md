# Patch-Based-DICOM-Image-and-Metadata-Compression

This algorithm is based on HiFiC(High-Fidelity Generative Image Compression). The original HiFiC software can be found in [link](https://github.com/tensorflow/compression/tree/master/models/hific)
---
## Download pretrained SAM model
[link]{https://drive.google.com/file/d/1LTJWX7cEV5KCML20nrysUROqFWc6ZCDb/view?usp=sharing}

## Usage of Patch-Based-DICOM-Image-and-Metadata-Compression

#### Compress a file
```
python tfci_sam.py compress [hific model] [path to dicom image] [path to output file]
```

#### Decompress a file
```
python tfci_sam.py decompress [path to compressed tfci file] [path to output file] --original_dicom_folder [path to original dicom file]
```

#### Compress a folder
```
python tfci_sam.py compress [hific model] [path to dicom folder] [path to output folder]
```

#### Decompress a file
```
python tfci_sam.py decompress [path to compressed tfci folder] [path to output folder] --original_dicom_folder [path to original dicom folder]
```
