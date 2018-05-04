# gpgpu_bootcamp
GPGPU and CUDA Tutorial for IceCube Bootcamp

Requirements
============
* GPU (duh)
* numba
* CUDA 8 (9 doesn't work well with numba yet)
* numpy
* ipython (optional)

Installation Instructions
=========================
The easiest and safest way to do this is to install conda (I prefer miniconda) first.
You can do this without admin priviliges and it turns out, for conda, this is preferred.
DON'T INSTALL WITH SUDO, even if you have sudo privileges.

https://conda.io/miniconda.html

Install the following packages:

$HOME/miniconda2/bin/conda install numba ipython cudatoolkit=8
(assuming the default installation of miniconda in $HOME)