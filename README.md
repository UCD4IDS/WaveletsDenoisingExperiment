# Denoising Experiment using Wavelet Transforms, Autocorrelation Wavelet Transforms, Stationary Wavelet Transforms
In this experiment, we observe the denoising capabilities of the different types of wavelet transforms on a group of signals. The following are the parameters we aim to understand:  
* Best threshold determination method: Individual vs Average vs Median of a set of best threshold values.
* Denoising method: [VisuShrink](https://www.jstor.org/stable/2291512?seq=1#metadata_info_tab_contents) vs [Threshold determination based on relative error curve](https://escholarship.org/content/qt0bv9t4c8/qt0bv9t4c8_noSplash_66d3d84d7c4f3146a80f5611e0214b1b.pdf)

## Authors
This experiment is conducted by Zeng Fung Liew and Shozen Dan under the supervision of Professor Naoki Saito at the University of California, Davis.

## Table of Contents
1. [Setup](#setup)
2. [Results](#results)
3. [Pluto notebook containing results and report](notebooks/denoising.jl)
4. Codes
    * [Denoising experiment](src/denoisingexperiments.jl)
    * [Denoising analysis](src/denoisinganalysis.jl)

## How to Open and Run Pluto Notebook <a name="setup"></a>
### Method 1 (**Recommended**): Opening notebook by cloning this repository
1. Clone the repository by typing the following:
```shell
git clone https://github.com/UCD4IDS/WaveletsDenoisingExperiment.git
```
2. Navigate to the WaveletsDenoisingExperiment directory and open up the Julia REPL.
3. Ensure Julia is working on the current directory. This can be checked using the following commands:
```julia
# shows the current working directory
julia> pwd() 

# change to the directory containing all the files from this repository. Eg:
julia> cd("C:/Users/USER/Documents/WaveletsDenoisingExperiment")
```
4. Enter the package manager in the REPL by typing `]`. The following should be observed:
```julia
(@v1.6) pkg> 
```
5. Activate the current environment by typing the following.   
Note: Steps 3-4 has to be done correctly for this step to work!
```julia
(@v1.6) pkg> activate ./notebooks
(@v1.6) pkg> instantiate
```  

6. Exit the package manager mode by hitting the backspace key. Then, type in the following commands:
```julia
julia> import Pluto; Pluto.run()
```

7. Pluto should open up in the default browser. Open up the file by keying in the file path.

### Method 2: Opening notebook directly without downloading any files from this repository
1. Open up the Julia REPL.
2. Manually install the required packages for running the notebooks. The list of required packages can be found in the [Project.toml](notebooks/Project.toml) file under the notebook directory.  
Install the packages in Julia using either the REPL or through the package manager. The package manager can be activated by hitting the `]` key. Example:
```julia
# install on REPL
julia> using Pkg; Pkg.add("Pluto")
# install on package manager
(@v1.6) pkg> add Pluto
```
3. Return to the REPL and type the command below. If you are currently at the package manager mode, you can return to the REPL by hitting the backspace key.
```julia
julia> import Pluto; Pluto.run()
```
4. Pluto should open up in the default browser. Copy-paste the following URL into the file path:  
[https://github.com/zengfung/WaveletsDenoisingExperiment/blob/master/notebooks/denoising.jl](https://github.com/zengfung/WaveletsDenoisingExperiment/blob/master/notebooks/denoising.jl)

**Note:** When opening the notebooks using this method, Julia automatically downloads the notebook into the `~/.julia/pluto_notebooks` folder in your local machine. You may want to delete them once you are done.


## Results <a name="results"></a>
![](figures/blocks.svg)
![](figures/bumps.svg)
![](figures/heavysine.svg)
![](figures/doppler.svg)
![](figures/quadchirp.svg)
![](figures/mishmash.svg)
