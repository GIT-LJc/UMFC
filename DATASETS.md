# How to install datasets

We suggest putting all datasets under the same folder (say `$DATA`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like

```
$DATA/
|–– domainnet/
|–– imagenet_ars/
```

## DomainNet
- Download the dataset from http://ai.bu.edu/M3SDA/.
- Extract the dataset to `$DATA/domainnet/images`. The file structure looks like

```
$DATA/
|–– domainnet/
    |–– images
        |–– clipart
        |–– infograph
        |–– ...
    |–– clusters
    |–– splits
```

## ImageNet-Variants
This dataset consists of three parts: ImageNet-Sketch, ImageNet-A, and ImageNet-R. The file structure looks like

```
$DATA/
|–– imagenet_ars/
    |–– images
        |–– imagenet-a
            |–– n01498041
            |–– ...
        |–– imagenet-r
            |–– n01443537
            |–– ...
        |–– imagenet-s
            |–– n01440764
            |–– ...
    |–– clusters
    |–– classnames.txt
```
### ImageNet-Sketch
- Download the dataset from https://github.com/HaohanWang/ImageNet-Sketch.
- Extract the dataset to `$DATA/imagenet_ars/images/imagenet-s`.

### ImageNet-A
- Download the dataset from https://github.com/hendrycks/natural-adv-examples and extract it to `$DATA/imagenet_ars/images/imagenet-a`.

### ImageNet-R
- Download the dataset from https://github.com/hendrycks/imagenet-r and extract it to `$DATA/imagenet_ars/images/imagenet-r`.
