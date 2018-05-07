Sample code to determine patrol stations given a map.
Coordinates are given in (x,y) pixel coordinates.
By default assumes grid world will be 32x32 pixels. There may be some collision due to downsampling.

A frame by frame plot of the locations will be saved in figures/.

# dependencies

For tensorflow, will need to find the correct link for your system at:
https://www.tensorflow.org/install/install_linux#the_url_of_the_tensorflow_python_package

```
conda create -n exp python=2.7
source activate exp
pip install opencv-python matplotlib
pip install scikit-fmm scipy 
pip install $TF_BINARY_URL
pip install tensorpack
```


# example

```
python compute_stations.py data/chicago4_45_2454_5673.png 
python compute_stations.py data/chicago4_45_2454_5673.png --size 64
```


