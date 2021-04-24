# Denoising Experiment using Wavelet Transforms, Autocorrelation Wavelet Transforms, Stationary Wavelet Transforms
In this experiment, we observe the denoising capabilities of the different types of wavelet transforms on a group of signals. The following are the parameters we aim to understand:  
* Best threshold determination method: Individual vs Average vs Median of a set of best threshold values.
* Denoising method: [VisuShrink](https://www.jstor.org/stable/2291512?seq=1#metadata_info_tab_contents) vs [Threshold determination based on relative error curve](https://escholarship.org/content/qt0bv9t4c8/qt0bv9t4c8_noSplash_66d3d84d7c4f3146a80f5611e0214b1b.pdf)

## Authors
This experiment is conducted by Zeng Fung Liew and Shozen Dan under the supervision of Professor Naoki Saito at the University of California, Davis.

## Table of Contents
1. [Code for denoising experiment](src/denoisingexperiments.jl)
2. [Code for denoising analysis](src/denoisinganalysis.jl)
3. [Pluto notebook containing results and report](notebooks/denoising.jl)

## How to Open and Run Pluto Notebook
### Method 1: Opening notebook directly without downloading any files
1. Open up the Julia REPL.
2. (Skip to step 3 if the Pluto package is already installed in Julia).   
Install Pluto in Julia either on the REPL or in the package manager. The package manager can be activated by hitting the `]` key.
```
# install on REPL
julia> using Pkg; Pkg.add("Pluto")
# install on package manager
(@v1.6) pkg> add Pluto
```
3. Return to the REPL and type the following command:
```
julia> import Pluto; Pluto.run()
```
4. Pluto should open up in the default browser. Copy-paste the following URL into the file path:  
[https://github.com/zengfung/WaveletsDenoisingExperiment/blob/master/notebooks/denoising.jl](https://github.com/zengfung/WaveletsDenoisingExperiment/blob/master/notebooks/denoising.jl)

### Method 2 (**Recommended**): Opening notebook after cloning this repository
1. Open up the Julia REPL.
2. Ensure Julia is working on the current directory. This can be checked using the following commands:
```
# shows the current working directory
julia> pwd() 

# change to the directory containing all the files from this repository. Eg:
julia> cd("C:/Users/USER/Documents/WaveletsDenoisingExperiment")
```
3. Enter the package manager in the REPL by typing `]`. The following should be observed:
```
(@v1.6) pkg> 
```
4. Activate the current environment by typing the following.   
Note: Step 2 has to be done correctly for this step to work!
```
(@v1.6) pkg> activate ./notebooks
(@v1.6) pkg> instantiate
```  

5. Exit the package manager mode by hitting the backspace key. Then, type in the following commands:
```
julia> import Pluto; Pluto.run()
```

6. Pluto should open up in the default browser. Open up the file by keying in the file path.
