# ct-based-diaphragm-function-evaluation

Convert Dicom data set to raw+mhd file
----------------

```shell
/Applications/ITK-SNAP.app/Contents/bin/c3d -dicom-series-list Batch_Series_INS_8023
```

```shell
/Applications/ITK-SNAP.app/Contents/bin/c3d -dicom-series-read Batch_Series_INS_8023 1.3.6.1.4.1.43276.1.3.20210611.1113419.1750.11904.16783.80231.000000512512 -output-multicomponent Batch_Series_INS_8023/output.mhd
```

Reference
----------------

https://sourceforge.net/p/c3d/git/ci/master/tree/doc/c3d.md#-dicom-series-list-list-image-series-in-a-dicom-directory

https://www.programmersought.com/article/81425633592/