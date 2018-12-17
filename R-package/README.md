# R wrapper for the IVIS algorithm

## Installation
R will install ivis into "ivis-supervised" conda environment. 

The easiest way to install ivis is using the `devtools` package:

```
devtools::install_github("beringresearch/ivis-supervised/R-package")
library(ivis_supervised)
install_ivis()
```

To set environment to tensorflow, add the following to your environment variables:
```
export KERAS_BACKEND=tensorflow
```

## Example
```
library(ivis_supervised)

xy <- ivis(iris[, 1:4], k = 3)
```
