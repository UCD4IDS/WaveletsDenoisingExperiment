### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ f9e001ae-0323-4283-9af1-1a8252b503e7
let
    import Pkg
    Pkg.activate(".")
	Pkg.instantiate()
end

# ╔═╡ 203fb8c8-4358-4908-b616-a691ce329c02
using 
	Wavelets, 
	WaveletsExt,
	LinearAlgebra, 
	Plots,
	Gadfly,
	DataFrames,
	PlutoUI,
	Statistics

# ╔═╡ 53c257e0-96ba-11eb-3615-8bfed63b2c18
md"# Denoising Experiment"

# ╔═╡ 78225642-d1bb-41ca-9b77-1b9ad1b5d9a0
md"""
## Introduction

Signal denoising is an important step in many signal processing and analysis related work as it helps reduce noise in the data while retaining important information. Throughout the years, many signal denoising algorithms have been developed. Thresholding is a simple yet effective technique of denoising signals and images. Here, we look into some thresholding methods with respect to different types of wavelet transforms.

The two main packages used in this experiment are Wavelets.jl and WaveletsExt.jl.

[Wavelets.jl](https://github.com/JuliaDSP/Wavelets.jl) is a very useful package for wavelet analysis in Julia, as it contains important preliminary tools such as wavelet constructors, wavelet transform, and thresholding functions.

[WaveletsExt.jl](https://github.com/UCD4IDS/WaveletsExt.jl) on the other hand, is an extension package to Wavelets.jl with added support for the stationary wavelet transforms (SWT), the autocorrelation wavelet transform (ACWT) and group-based best basis algorithms such as the joint best basis (JBB) and least-statistically dependent basis (LSDB).

In this experiment, we will compare and contrast the signal denoising strengths and weaknesses of different wavelet transform and threshold selection methods. Specifically, given a wavelet, a group of signals of length $2^L$ and a threshold method (eg. Hard thresholding), we will compare the denoising ability between:
* type of wavelet transform:
  * regular discrete wavelet transforms.
  * regular wavelet packet decomposition up to $L/2$ levels.
  * regular wavelet packet decomposition up to $L$ levels.
  * regular wavelet packet best basis transforms.
  * regular wavelet packet joint best basis transforms.
  * regular wavelet packet least-statistically dependent basis transforms.
  * autocorrelation discrete wavelet transforms.
  * autocorrelation wavelet packet decomposition up to $L/2$ levels.
  * autocorrelation wavelet packet decomposition up to $L$ levels.
  * autocorrelation wavelet packet best basis transforms.
  * autocorrelation wavelet packet joint best basis transforms.
  * autocorrelation wavelet packet least-statistically dependent basis transforms.
  * stationary discrete wavelet transforms.
  * stationary wavelet packet decomposition up to $L/2$ levels.
  * stationary wavelet packet decomposition up to $L$ levels.
  * stationary wavelet packet best basis transforms.
  * stationary wavelet packet joint best basis transforms.
  * stationary wavelet packet least-statistically dependent basis transforms.
* type of threshold selection algorithm:
  * VisuShrink developed by D. Donoho and I. Johnstone.
  * RelErrorShrink used in N. Saito and J. Irion in their paper ["Efficient Approximation and Denoising of Graph Signals Using the Multiscale Best Basis Dictionaries"](https://escholarship.org/content/qt0bv9t4c8/qt0bv9t4c8_noSplash_66d3d84d7c4f3146a80f5611e0214b1b.pdf).
  * SureShrink developed by D. Donoho and I. Johnstone.
* selection of the best threshold values:
  * using individual best threshold values selected from the threshold selection algorithm.
  * using the average of the best threshold values selected from the threshold selection algorithm.
  * using the median of the best threshold values selected from the threshold selection algorithm.
"""

# ╔═╡ ce4bf94e-edef-40c2-8ac5-b741e47a1759
md"""
## Activate environment

This is used for activating the project environment and is specifically catered for users who cloned the notebook repository. 

**Do not change anything in the following block of code.**
"""

# ╔═╡ 85c26ead-f043-42c8-8245-58c8d03a963d
md"""
## Import libraries

**Do not change anything in the following block of code.**
"""

# ╔═╡ 46e30602-a850-4175-b4bf-b4ef4b5359aa
md"""
## Exploratory Data Analysis

In this section, we will have a quick look and understanding of what our test signals look like, and how thresholding visually results in signals that better resemble the original signals.
"""

# ╔═╡ b0a28618-7cda-4c05-83d2-b54bbca3f9b5
md"""
**Select** a test function. These signals are obtained from D. Donoho and I. Johnstone in ["Adapting to Unknown Smoothness via Wavelet Shrinkage"](http://statweb.stanford.edu/~imj/WEBLIST/1995/ausws.pdf) Preprint Stanford, January 93, p 27-28.

$(@bind signal_name_test Select(
	["blocks", "bumps", "heavy_sine", "doppler", "quadchirp", "mishmash"],
	default = "doppler"
))
"""

# ╔═╡ bb147649-36bf-4a82-95bf-2d5080873028
md"""
**Select** which type of wavelet to use: $(@bind wavelet_type_test Select(
	["WT.haar", 
	"WT.db1", "WT.db2", "WT.db3", "WT.db4", "WT.db5", 
	"WT.db6", "WT.db7", "WT.db8", "WT.db9", "WT.db10",
	"WT.coif2", "WT.coif4", "WT.coif6", "WT.coif8",
	"WT.sym4", "WT.sym5", "WT.sym6", "WT.sym7", "WT.sym8", "WT.sym9", "WT.sym10",
	"WT.batt2", "WT.batt4", "WT.batt6"],
	default = "WT.haar"
))
"""

# ╔═╡ ff1bfee3-2f30-4d15-9f3a-e3f422e67d72
md"""
**Select** the value $L$ for signal of length $2^L$

$(@bind max_dec_level_test Slider(6:1:10, default=8, show_value=true))
"""

# ╔═╡ 458a5a1e-c453-4199-befe-2bf4db6825ae
md"""
**Adjust** the magnitude of Gaussian noise

$(@bind noise_size_test Slider(0:0.01:1, default=0.3, show_value=true))
"""

# ╔═╡ 7c0dba17-baf3-4b9c-b1c5-486f7e4515f4
md"""
The `addnoise` function will add Gaussian noise that is proportional to the total energy of the signal.

**Do not change anything in the following block of code.**
"""

# ╔═╡ a184ae65-7947-4ffb-b751-8b6e97a8608b
function addnoise(x::AbstractArray{<:Number,1}, s::Real=0.1)
	ϵ = randn(length(x))
	ϵ = ϵ/norm(ϵ)
	y = x + ϵ * s * norm(x)
	return y
end;

# ╔═╡ faa0a4ea-7849-408a-ac0f-c5cca8761aee
md"""
The following plots show us the decomposed noisy signal using autocorrelation discrete wavelet transform, stationary discrete wavelet transform, and the regular discrete wavelet transform. 

For ACWT and SWT, we can see that the signals are decomposed well, and that the bottom signals look similar to the original signal.

**Note:** As the regular wavelet transform is non-redundant, ie. downsampling takes place with a factor of 2 at every level, one cannot conclude that the regular transform of noisy signal is doing a bad job. The wavelet coefficients for the regular DWT were padded and extended such that they obtain the length of the original signal. This is done strictly for comparison and illustration purposes only.
"""

# ╔═╡ 356d75f4-6cc1-4062-8ef6-3cc6a9c2d9a7
md"Plot a histogram of wavelet coefficients:"

# ╔═╡ 9eae2f5c-1f47-43d8-a7ec-0767e50e6a9b
md"""
**Select** threshold type

$(@bind th_method_test Radio(["Hard", "Soft"], default = "Hard"))
"""

# ╔═╡ e214b298-04d8-473f-b56e-5f446374078c
md"""
Now that we have analyzed the properties of the signals and the effects of noise on them, let's observe the denoised/thresholded signals below.

To do so, we will have to:
1. Threshold the wavelet coefficients.
2. Reconstruct the signal based on the thresholded coefficients.
"""

# ╔═╡ 11e63c9a-6124-4122-9a86-ceed926d25d2
md"## Experimental Data Setup"

# ╔═╡ d881753b-0432-451b-8de0-38a0b4b4382a
md"**Autorun**: $(@bind autorun CheckBox())

**Note:** Disable before updating parameters! "

# ╔═╡ 8055194b-2e46-4d18-81c0-0c52bc3eb233
md"""
**Select** a test function $(@bind signal_name Select(
	[
		"blocks" => "Blocks", 
		"bumps" => "Bumps", 
		"heavy_sine" => "Heavy sine", 
		"doppler" => "Doppler", 
		"quadchirp" => "Quadchirp", 
		"mishmash" => "Mishmash"
	],
	default = "blocks"
))

"""

# ╔═╡ f6277c19-1989-449e-96ba-6f81db68c76b
md"""
**Select** the value $L$ for signal of length $2^L$

*Warning: The computation starts to take extremely long when setting $L \geq 9$.*

$(@bind max_dec_level Slider(6:1:10, default=7, show_value=true))
"""

# ╔═╡ 56ee2c61-d83c-4d76-890a-a9bd0d65cee5
md"**Adjust** the slider to add Gaussian noise to the test signal"

# ╔═╡ c50ac92e-3684-4d0a-a80d-4ee9d74ec992
@bind noise_size Slider(0:0.01:1, default=0.3, show_value=true)

# ╔═╡ e0a96592-5e77-4c29-9744-31369eea8147
md"""
**Select** which type of wavelet to use: $(@bind wavelet_type Select(
	["WT.haar", 
	"WT.db1", "WT.db2", "WT.db3", "WT.db4", "WT.db5", 
	"WT.db6", "WT.db7", "WT.db8", "WT.db9", "WT.db10",
	"WT.coif2", "WT.coif4", "WT.coif6", "WT.coif8",
	"WT.sym4", "WT.sym5", "WT.sym6", "WT.sym7", "WT.sym8", "WT.sym9", "WT.sym10",
	"WT.batt2", "WT.batt4", "WT.batt6"],
	default = "WT.haar"
))
"""

# ╔═╡ 851b04bb-e382-4a0a-98f6-7d4b983ca5ab
begin
	wt_test = wavelet(eval(Meta.parse(wavelet_type)))
	
	x_test = generatesignals(Symbol(signal_name_test), max_dec_level_test)
	p1_test = Plots.plot(x_test, ylim = (minimum(x_test)-1,maximum(x_test)+1), label = "Original signal", title = "Original vs Noisy")
	x_noisy_test = addnoise(x_test, noise_size_test)
	Plots.plot!(x_noisy_test, label = "Noisy signal");
	
	y_test = acwt(x_noisy_test, wt_test, max_dec_level_test÷2);
	p2_test = WaveletsExt.wiggle(y_test, Overlap=false)
	Plots.plot!(p2_test, title = "Autocorrelation Transform of Noisy Signal")
	
	z_test = sdwt(x_noisy_test, wt_test, max_dec_level_test÷2);
	p3_test = WaveletsExt.wiggle(z_test, Overlap=false)
	Plots.plot!(p3_test, title = "Stationary Transform of Noisy Signal")
	
	function extend_signal(x::AbstractVector{T}, l::Int) where T
		n = length(x)					# signal length
		L = maxtransformlevels(x)		# max transform levels of x
		y = Array{T,2}(undef, (n,l+1))
		lv = l 							# level of bottom left node
		nₙ = nodelength(n,l)			# length of bottom left node
		y[:,1] = repeat(x[1:nₙ], inner=1<<lv)
		st = nₙ+1
		for i in 2:(l+1)
			rng = st:(st+nₙ-1)
			y[:,i] = repeat(x[rng], inner=1<<lv)
			lv -= 1
			st += nₙ
			nₙ *= 2
		end
		return y
	end
	
	w_test = dwt(x_noisy_test, wt_test, max_dec_level_test÷2);
	we_test = extend_signal(w_test, max_dec_level_test÷2)
	p4_test = WaveletsExt.wiggle(we_test, Overlap=false)
	Plots.plot!(p4_test, title = "Regular Transform of Noisy Signal")
	
	Plots.plot(p1_test, p2_test, p3_test, p4_test, layout = (4,1), size=(600, 800))
end

# ╔═╡ 341c2551-b625-47a0-9163-3c0c0e7d4e13
Plots.histogram(vec(abs.(y_test)), legend = false)

# ╔═╡ dbed8579-afa4-4a4d-b0bb-bd34877fa272
md"""
Before we dig deeper into threshold value selection, let's understand what thresholding does to a noisy signal by playing with the threshold slider below.

**Select** a threshold value

$(@bind th_test Slider(0:0.01:maximum(y_test), default = 0, show_value = true))
"""

# ╔═╡ e49e76e6-7018-4f49-a189-d2fae7df956d
begin
	ŷ_test = copy(y_test)
	ẑ_test = copy(z_test)
	ŵ_test = copy(w_test)
	if th_method_test == "Hard"
		threshold!(ŷ_test, HardTH(), th_test);
		threshold!(ẑ_test, HardTH(), th_test);
		threshold!(ŵ_test, HardTH(), th_test);
	else
		threshold!(ŷ_test, SoftTH(), th_test);
		threshold!(ẑ_test, SoftTH(), th_test);
		threshold!(ŵ_test, SoftTH(), th_test);
	end
end;

# ╔═╡ 9b4ef541-9a36-4bc0-8654-10ab0a4e63b3
begin
	# reconstruction
	r1_test = iacwt(ŷ_test)
	r2_test = isdwt(ẑ_test, wt_test)
	r3_test = idwt(ŵ_test, wt_test, max_dec_level_test÷2)
	# plot original vs ACWT-denoised
	d1_test = Plots.plot(x_test, label = "original", lc = "black", lw=2)
	Plots.plot!(d1_test, r1_test, label = "ACWT denoised", lc="green", lw=1.5)
	Plots.plot!(d1_test, x_noisy_test, label = "noisy", lc = "gray", la = 0.8)
	# plot original vs SDWT-denoised
	d2_test = Plots.plot(x_test, label = "original", lc = "black", lw=2)
	Plots.plot!(d2_test, r2_test, label = "SWT denoised", lc="green", lw=1.5)
	Plots.plot!(d2_test, x_noisy_test, label = "noisy", lc = "gray", la = 0.8)
	# plot original vs DWT-denoised
	d3_test = Plots.plot(x_test, label = "original", lc = "black", lw=2)
	Plots.plot!(d3_test, r3_test, label = "DWT denoised", lc="green", lw=1.5)
	Plots.plot!(d3_test, x_noisy_test, label = "noisy", lc = "gray", la = 0.8)
	# display all plots
	Plots.plot(d1_test, d2_test, d3_test, layout = (3,1), size=(600, 600))
end

# ╔═╡ 4669be94-6c4c-42e2-b9d9-2dc98f1bdaea
md"""
Here are some metrics to determine how well our signal thresholding fared:

* Relative 2-norm between original signal and denoised signal (smaller value indicates better result)
| Signal | Relative 2-norm |
| :---: | :---: |
| noisy | $(round(relativenorm(x_noisy_test, x_test), digits = 4)) |
| ACWT denoised | $(round(relativenorm(r1_test, x_test), digits = 4)) |
| SWT denoised | $(round(relativenorm(r2_test, x_test), digits = 4)) |
| DWT denoised | $(round(relativenorm(r3_test, x_test), digits = 4)) |

* PSNR between original signal and denoised signal (larger value indicates better result)
| Signal | PSNR |
| :---: | :---: |
| noisy | $(round(psnr(x_noisy_test, x_test), digits = 4)) |
| ACWT denoised | $(round(psnr(r1_test, x_test), digits = 4)) |
| SWT denoised | $(round(psnr(r2_test, x_test), digits = 4)) |
| DWT denoised | $(round(psnr(r3_test, x_test), digits = 4)) |

* SSIM between original signal and denoised signal (larger value indicates better result)
| Signal | SSIM |
| :---: | :---: |
| noisy | $(round(ssim(x_noisy_test, x_test), digits = 4)) |
| ACWT denoised | $(round(ssim(r1_test, x_test), digits = 4)) |
| SWT denoised | $(round(ssim(r2_test, x_test), digits = 4)) |
| DWT denoised | $(round(ssim(r3_test, x_test), digits = 4)) |
"""

# ╔═╡ c178527f-96a4-4ac7-bb0c-38b73b38c45b
md"""
**Select** which type of thresholding to use: $(@bind threshold_method Select(
	["HardTH()" => "Hard",
	"SoftTH()" => "Soft",
	"SteinTH()" => "Stein"],
	default = "HardTH()"
))
"""

# ╔═╡ ef3e7b66-fba0-467a-8a73-c9bf31fadbe3
md"""
**Key in** the sample size: $(@bind ss TextField((5,1), default="10"))
"""

# ╔═╡ cd9e259e-8bb3-497b-ac7f-f89a003c8032
begin
	x = generatesignals(Symbol(signal_name), max_dec_level)
	p3 = Plots.plot(x, ylim = (minimum(x)-1,maximum(x)+1), label = "Original signal")
	x_noisy = addnoise(x, noise_size)
	Plots.plot!(x_noisy, label = "Noisy signal");
end

# ╔═╡ 3246e8b5-251f-4398-b21c-397341f2542e
md"**Preparing** the data
* Generate a set of original signals $\rightarrow$ `X₀`
* Add noise to original signals $\rightarrow$ `X`
"

# ╔═╡ 82e713f8-c870-43d2-a849-e3b401b00459
begin
	if autorun
		samplesize = parse(Int64, ss)
		X₀ = duplicatesignals(x, samplesize, 2)
		X = hcat([addnoise(X₀[:,i], noise_size) for i in axes(X₀,2)]...)
	end
end;

# ╔═╡ 6b02c425-39b9-467f-9406-3e9096873af4
begin
	if autorun
		wt = wavelet(eval(Meta.parse(wavelet_type)))
		th = eval(Meta.parse(threshold_method))
		vs_dnt = VisuShrink(256, th)
		res_dnt = RelErrorShrink(th)
		# define variables to store results
		Y = Dict{String, AbstractArray}() # Decompositions
		X̂ = Dict{String, AbstractArray}() 
		T = Dict{String, AbstractArray}() # Trees
		σ₁= Dict{String, AbstractArray}() # Noise estimates for VisuShrink
		σ₂= Dict{String, AbstractArray}() # Noise estimates for RelErrorShrink
		time = Dict{String, AbstractFloat}()
		mean_psnr = Dict{String, AbstractFloat}()
		mean_ssim = Dict{String, AbstractFloat}()
		results = DataFrame(
			transform = ["None"], 
			threshold = ["None"],
			shrinking = ["None"],
			selection = ["None"],
			time = 0.0,
			PSNR = mean([psnr(X[:,i], X₀[:,i]) for i in axes(X,2)]),
			SSIM = mean([ssim(X[:,i], X₀[:,i]) for i in axes(X,2)])
		)
	end
end

# ╔═╡ 95081e88-a623-4e91-99c1-8b254b366dac
md"Once you are satisfied with the selected parameters, enabling the **autorun** option above will prompt the experiment to run."

# ╔═╡ 126c41e7-dd65-46c6-8c5b-2439f5624fd5
md"# Non-Redundant Transforms"

# ╔═╡ 17bdc97a-4a0b-4931-a5c6-866f0c814601
md"### 1. Discrete Wavelet Transform"

# ╔═╡ 01e43234-2194-451d-9010-176aa4799fdb
begin
	if autorun
		# Visushrink
		Y["DWT"] = cat([dwt(X[:,i], wt) for i in axes(X,2)]..., dims=2)
		σ₁["DWT"] = [noisest(Y["DWT"][:,i], false) for i in axes(X,2)]
		σ₂["DWT"] = [relerrorthreshold(Y["DWT"][:,i], false) for i in axes(X,2)]
		## bestTH = individual
		X̂["DWT_vs_ind"], time["DWT_vs_ind"] = @timed denoiseall(
			Y["DWT"], 
			:dwt, 
			wt, 
			L=max_dec_level, 
			estnoise=σ₁["DWT"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["DWT_vs_ind"] = mean(
			[psnr(X̂["DWT_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["DWT_vs_ind"] = mean(
			[ssim(X̂["DWT_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"DWT", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["DWT_vs_ind"], 
				mean_psnr["DWT_vs_ind"],
				mean_ssim["DWT_vs_ind"]
			]
		)
		## bestTH = average
		X̂["DWT_vs_avg"], time["DWT_vs_avg"] = @timed denoiseall(
			Y["DWT"], 
			:dwt, 
			wt, 
			L=max_dec_level, 
			estnoise=σ₁["DWT"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["DWT_vs_avg"] = mean(
			[psnr(X̂["DWT_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["DWT_vs_avg"] = mean(
			[ssim(X̂["DWT_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"DWT", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["DWT_vs_avg"], 
				mean_psnr["DWT_vs_avg"],
				mean_ssim["DWT_vs_avg"]
			]
		)
		## bestTH = median
		X̂["DWT_vs_med"], time["DWT_vs_med"] = @timed denoiseall(
			Y["DWT"], 
			:dwt, 
			wt, 
			L=max_dec_level, 
			estnoise=σ₁["DWT"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["DWT_vs_med"] = mean(
			[psnr(X̂["DWT_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["DWT_vs_med"] = mean(
			[ssim(X̂["DWT_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"DWT", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["DWT_vs_med"], 
				mean_psnr["DWT_vs_med"],
				mean_ssim["DWT_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["DWT_res_ind"], time["DWT_res_ind"] = @timed denoiseall(
			Y["DWT"], 
			:dwt, 
			wt, 
			L=max_dec_level, 
			estnoise=σ₂["DWT"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["DWT_res_ind"] = mean(
			[psnr(X̂["DWT_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["DWT_res_ind"] = mean(
			[ssim(X̂["DWT_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"DWT", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["DWT_res_ind"], 
				mean_psnr["DWT_res_ind"],
				mean_ssim["DWT_res_ind"]
			]
		)
		## bestTH = average
		X̂["DWT_res_avg"], time["DWT_res_avg"] = @timed denoiseall(
			Y["DWT"], 
			:dwt, 
			wt, 
			L=max_dec_level, 
			estnoise=σ₂["DWT"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["DWT_res_avg"] = mean(
			[psnr(X̂["DWT_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["DWT_res_avg"] = mean(
			[ssim(X̂["DWT_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"DWT", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["DWT_res_avg"], 
				mean_psnr["DWT_res_avg"],
				mean_ssim["DWT_res_avg"]
			]
		)
		## bestTH = median
		X̂["DWT_res_med"], time["DWT_res_med"] = @timed denoiseall(
			Y["DWT"], 
			:dwt, 
			wt, 
			L=max_dec_level, 
			estnoise=σ₂["DWT"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["DWT_res_med"] = mean(
			[psnr(X̂["DWT_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["DWT_res_med"] = mean(
			[ssim(X̂["DWT_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"DWT", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["DWT_res_med"], 
				mean_psnr["DWT_res_med"],
				mean_ssim["DWT_res_med"]
			]
		)
	end
end;

# ╔═╡ ae8059bd-5b5b-4ff2-a6f0-5ce672bdd54d
md"### 2. Wavelet Packet Transform - Level $(max_dec_level÷2)"

# ╔═╡ cf55c5cb-ead6-40b6-896a-8f7e01613a46
begin
	if autorun
		# Visushrink
		Y["WPT$(max_dec_level÷2)"] = cat(
			[wpt(X[:,i], wt, max_dec_level÷2) for i in axes(X,2)]..., dims=2
		)
		T["WPT$(max_dec_level÷2)"] = maketree(
			1<<max_dec_level, max_dec_level÷2, :full
		)
		σ₁["WPT$(max_dec_level÷2)"] = [
			noisest(
				Y["WPT$(max_dec_level÷2)"][:,i], false, T["WPT$(max_dec_level÷2)"]
			) for i in axes(X,2)
		]
		σ₂["WPT$(max_dec_level÷2)"] = [
			relerrorthreshold(
				Y["WPT$(max_dec_level÷2)"][:,i], false, T["WPT$(max_dec_level÷2)"]
			) for i in axes(X,2)
		]
		## bestTH = individual
		X̂["WPT$(max_dec_level÷2)_vs_ind"], time["WPT$(max_dec_level÷2)_vs_ind"] = 
		@timed denoiseall(
			Y["WPT$(max_dec_level÷2)"], 
			:wpt, 
			wt, 
			tree=T["WPT$(max_dec_level÷2)"], 
			estnoise=σ₁["WPT$(max_dec_level÷2)"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["WPT$(max_dec_level÷2)_vs_ind"] = mean(
			[psnr(X̂["WPT$(max_dec_level÷2)_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT$(max_dec_level÷2)_vs_ind"] = mean(
			[ssim(X̂["WPT$(max_dec_level÷2)_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L$(max_dec_level÷2)", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["WPT$(max_dec_level÷2)_vs_ind"], 
				mean_psnr["WPT$(max_dec_level÷2)_vs_ind"],
				mean_ssim["WPT$(max_dec_level÷2)_vs_ind"]
			]
		)
		## bestTH = average
		X̂["WPT$(max_dec_level÷2)_vs_avg"], time["WPT$(max_dec_level÷2)_vs_avg"] = 
		@timed denoiseall(
			Y["WPT$(max_dec_level÷2)"], 
			:wpt, 
			wt, 
			tree=T["WPT$(max_dec_level÷2)"], 
			estnoise=σ₁["WPT$(max_dec_level÷2)"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["WPT$(max_dec_level÷2)_vs_avg"] = mean(
			[psnr(X̂["WPT$(max_dec_level÷2)_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT$(max_dec_level÷2)_vs_avg"] = mean(
			[ssim(X̂["WPT$(max_dec_level÷2)_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L$(max_dec_level÷2)", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["WPT$(max_dec_level÷2)_vs_avg"], 
				mean_psnr["WPT$(max_dec_level÷2)_vs_avg"],
				mean_ssim["WPT$(max_dec_level÷2)_vs_avg"]
			]
		)
		## bestTH = median
		X̂["WPT$(max_dec_level÷2)_vs_med"], time["WPT$(max_dec_level÷2)_vs_med"] =
		@timed denoiseall(
			Y["WPT$(max_dec_level÷2)"], 
			:wpt, 
			wt, 
			tree=T["WPT$(max_dec_level÷2)"], 
			estnoise=σ₁["WPT$(max_dec_level÷2)"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["WPT$(max_dec_level÷2)_vs_med"] = mean(
			[psnr(X̂["WPT$(max_dec_level÷2)_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT$(max_dec_level÷2)_vs_med"] = mean(
			[ssim(X̂["WPT$(max_dec_level÷2)_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L$(max_dec_level÷2)", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["WPT$(max_dec_level÷2)_vs_med"], 
				mean_psnr["WPT$(max_dec_level÷2)_vs_med"],
				mean_ssim["WPT$(max_dec_level÷2)_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["WPT$(max_dec_level÷2)_res_ind"], time["WPT$(max_dec_level÷2)_res_ind"] =
		@timed denoiseall(
			Y["WPT$(max_dec_level÷2)"], 
			:wpt, 
			wt, 
			tree=T["WPT$(max_dec_level÷2)"], 
			estnoise=σ₂["WPT$(max_dec_level÷2)"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["WPT$(max_dec_level÷2)_res_ind"] = mean(
			[psnr(X̂["WPT$(max_dec_level÷2)_res_ind"][:,i],X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT$(max_dec_level÷2)_res_ind"] = mean(
			[ssim(X̂["WPT$(max_dec_level÷2)_res_ind"][:,i],X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L$(max_dec_level÷2)", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["WPT$(max_dec_level÷2)_res_ind"], 
				mean_psnr["WPT$(max_dec_level÷2)_res_ind"],
				mean_ssim["WPT$(max_dec_level÷2)_res_ind"]
			]
		)
		## bestTH = average
		X̂["WPT$(max_dec_level÷2)_res_avg"], time["WPT$(max_dec_level÷2)_res_avg"] =
		@timed denoiseall(
			Y["WPT$(max_dec_level÷2)"], 
			:wpt, 
			wt, 
			tree=T["WPT$(max_dec_level÷2)"], 
			estnoise=σ₂["WPT$(max_dec_level÷2)"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["WPT$(max_dec_level÷2)_res_avg"] = mean(
			[psnr(X̂["WPT$(max_dec_level÷2)_res_avg"][:,i],X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT$(max_dec_level÷2)_res_avg"] = mean(
			[ssim(X̂["WPT$(max_dec_level÷2)_res_avg"][:,i],X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L$(max_dec_level÷2)", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["WPT$(max_dec_level÷2)_res_avg"], 
				mean_psnr["WPT$(max_dec_level÷2)_res_avg"],
				mean_ssim["WPT$(max_dec_level÷2)_res_avg"]
			]
		)
		## bestTH = median
		X̂["WPT$(max_dec_level÷2)_res_med"], time["WPT$(max_dec_level÷2)_res_med"] =
		@timed denoiseall(
			Y["WPT$(max_dec_level÷2)"], 
			:wpt, 
			wt, 
			tree=T["WPT$(max_dec_level÷2)"], 
			estnoise=σ₂["WPT$(max_dec_level÷2)"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["WPT$(max_dec_level÷2)_res_med"] = mean(
			[psnr(X̂["WPT$(max_dec_level÷2)_res_med"][:,i],X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT$(max_dec_level÷2)_res_med"] = mean(
			[ssim(X̂["WPT$(max_dec_level÷2)_res_med"][:,i],X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L$(max_dec_level÷2)", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["WPT$(max_dec_level÷2)_res_med"], 
				mean_psnr["WPT$(max_dec_level÷2)_res_med"],
				mean_ssim["WPT$(max_dec_level÷2)_res_med"]
			]
		)
	end
end;

# ╔═╡ c95ebbed-3d9a-4be2-943b-08c86923ad89
md"### 3. Wavelet Packet Transform - Level $(max_dec_level)"

# ╔═╡ 61d745d8-5c74-479b-9698-cd50bb68b3c7
begin
	if autorun
		# Visushrink
		Y["WPT$(max_dec_level)"] = cat(
			[wpt(X[:,i], wt,max_dec_level) for i in axes(X,2)]..., dims=2
		)
		T["WPT$(max_dec_level)"] = maketree(1<<max_dec_level, max_dec_level, :full)
		σ₁["WPT$(max_dec_level)"] = [
			noisest(
				Y["WPT$(max_dec_level)"][:,i], false, T["WPT$(max_dec_level)"]
			) for i in axes(X,2)
		]
		σ₂["WPT$(max_dec_level)"] = [
			relerrorthreshold(
				Y["WPT$(max_dec_level)"][:,i], false, T["WPT$(max_dec_level)"]
			) for i in axes(X,2)
		]
		## bestTH = individual
		X̂["WPT$(max_dec_level)_vs_ind"], time["WPT$(max_dec_level)_vs_ind"] = 
		@timed denoiseall(
			Y["WPT$(max_dec_level)"], 
			:wpt, 
			wt, 
			tree=T["WPT$(max_dec_level)"], 
			estnoise=σ₁["WPT$(max_dec_level)"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["WPT$(max_dec_level)_vs_ind"] = mean(
			[psnr(X̂["WPT$(max_dec_level)_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT$(max_dec_level)_vs_ind"] = mean(
			[ssim(X̂["WPT$(max_dec_level)_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L$(max_dec_level)", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["WPT$(max_dec_level)_vs_ind"], 
				mean_psnr["WPT$(max_dec_level)_vs_ind"],
				mean_ssim["WPT$(max_dec_level)_vs_ind"]
			]
		)
		## bestTH = average
		X̂["WPT$(max_dec_level)_vs_avg"], time["WPT$(max_dec_level)_vs_avg"] = 
		@timed denoiseall(
			Y["WPT$(max_dec_level)"], 
			:wpt, 
			wt, 
			tree=T["WPT$(max_dec_level)"], 
			estnoise=σ₁["WPT$(max_dec_level)"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["WPT$(max_dec_level)_vs_avg"] = mean(
			[psnr(X̂["WPT$(max_dec_level)_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT$(max_dec_level)_vs_avg"] = mean(
			[ssim(X̂["WPT$(max_dec_level)_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L$(max_dec_level)", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["WPT$(max_dec_level)_vs_avg"], 
				mean_psnr["WPT$(max_dec_level)_vs_avg"],
				mean_ssim["WPT$(max_dec_level)_vs_avg"]
			]
		)
		## bestTH = median
		X̂["WPT$(max_dec_level)_vs_med"], time["WPT$(max_dec_level)_vs_med"] = 
		@timed denoiseall(
			Y["WPT$(max_dec_level)"], 
			:wpt, 
			wt, 
			tree=T["WPT$(max_dec_level)"], 
			estnoise=σ₁["WPT$(max_dec_level)"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["WPT$(max_dec_level)_vs_med"] = mean(
			[psnr(X̂["WPT$(max_dec_level)_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT$(max_dec_level)_vs_med"] = mean(
			[ssim(X̂["WPT$(max_dec_level)_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L$(max_dec_level)", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["WPT$(max_dec_level)_vs_med"], 
				mean_psnr["WPT$(max_dec_level)_vs_med"],
				mean_ssim["WPT$(max_dec_level)_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["WPT$(max_dec_level)_res_ind"], time["WPT$(max_dec_level)_res_ind"] = 
		@timed denoiseall(
			Y["WPT$(max_dec_level)"], 
			:wpt, 
			wt, 
			tree=T["WPT$(max_dec_level)"], 
			estnoise=σ₂["WPT$(max_dec_level)"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["WPT$(max_dec_level)_res_ind"] = mean(
			[psnr(X̂["WPT$(max_dec_level)_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT$(max_dec_level)_res_ind"] = mean(
			[ssim(X̂["WPT$(max_dec_level)_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L$(max_dec_level)", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["WPT$(max_dec_level)_res_ind"], 
				mean_psnr["WPT$(max_dec_level)_res_ind"],
				mean_ssim["WPT$(max_dec_level)_res_ind"]
			]
		)
		## bestTH = average
		X̂["WPT$(max_dec_level)_res_avg"], time["WPT$(max_dec_level)_res_avg"] = 
		@timed denoiseall(
			Y["WPT$(max_dec_level)"], 
			:wpt, 
			wt, 
			tree=T["WPT$(max_dec_level)"], 
			estnoise=σ₂["WPT$(max_dec_level)"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["WPT$(max_dec_level)_res_avg"] = mean(
			[psnr(X̂["WPT$(max_dec_level)_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT$(max_dec_level)_res_avg"] = mean(
			[ssim(X̂["WPT$(max_dec_level)_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L$(max_dec_level)", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["WPT$(max_dec_level)_res_avg"], 
				mean_psnr["WPT$(max_dec_level)_res_avg"],
				mean_ssim["WPT$(max_dec_level)_res_avg"]
			]
		)
		## bestTH = median
		X̂["WPT$(max_dec_level)_res_med"], time["WPT$(max_dec_level)_res_med"] = @timed denoiseall(
			Y["WPT$(max_dec_level)"], 
			:wpt, 
			wt, 
			tree=T["WPT$(max_dec_level)"], 
			estnoise=σ₂["WPT$(max_dec_level)"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["WPT$(max_dec_level)_res_med"] = mean(
			[psnr(X̂["WPT$(max_dec_level)_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT$(max_dec_level)_res_med"] = mean(
			[ssim(X̂["WPT$(max_dec_level)_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L$(max_dec_level)", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["WPT$(max_dec_level)_res_med"], 
				mean_psnr["WPT$(max_dec_level)_res_med"],
				mean_ssim["WPT$(max_dec_level)_res_med"]
			]
		)
	end
end;

# ╔═╡ e5738623-4ea8-4866-a1da-7e849960f4e0
md"### 4. Wavelet Packet Transform - Best Basis"

# ╔═╡ c34c1f0c-7cfd-404b-aa59-d4bb6aa9628f
begin
	if autorun
		# Visushrink
		Y["WPD"] = cat([wpd(X[:,i], wt) for i in axes(X,2)]..., dims=3)
		wpt_bt = bestbasistree(Y["WPD"], BB())
		Y["WPT-BT"] = cat(
			[wpt(X[:,i], wt, wpt_bt[:,i]) for i in axes(X,2)]..., dims=2
		)
		σ₁["WPT-BT"]= [
			noisest(Y["WPT-BT"][:,i], false, wpt_bt[:,i]) for i in axes(X,2)
		]
		σ₂["WPT-BT"]= [
			relerrorthreshold(
				Y["WPT-BT"][:,i], false, wpt_bt[:,i]
			) for i in axes(X,2)
		]
		## bestTH = individual
		X̂["WPT-BT_vs_ind"], time["WPT-BT_vs_ind"] = @timed hcat([
			denoise(
				Y["WPT-BT"][:,i], 
				:wpt, 
				wt, 
				tree=wpt_bt[:,i], 
				dnt=vs_dnt, 
				estnoise=σ₁["WPT-BT"][i]
			) for i in 1:samplesize
		]...)
		mean_psnr["WPT-BT_vs_ind"] = mean(
			[psnr(X̂["WPT-BT_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT-BT_vs_ind"] = mean(
			[ssim(X̂["WPT-BT_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-BT", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["WPT-BT_vs_ind"], 
				mean_psnr["WPT-BT_vs_ind"],
				mean_ssim["WPT-BT_vs_ind"]
			]
		)
		## bestTH = average
		X̂["WPT-BT_vs_avg"], time["WPT-BT_vs_avg"] = @timed hcat([
			denoise(
				Y["WPT-BT"][:,i], 
				:wpt, 
				wt, 
				tree=wpt_bt[:,i], 
				dnt=vs_dnt, 
				estnoise=mean(σ₁["WPT-BT"])
			) for i in 1:samplesize
		]...)
		mean_psnr["WPT-BT_vs_avg"] = mean(
			[psnr(X̂["WPT-BT_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT-BT_vs_avg"] = mean(
			[ssim(X̂["WPT-BT_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-BT", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["WPT-BT_vs_avg"], 
				mean_psnr["WPT-BT_vs_avg"],
				mean_ssim["WPT-BT_vs_avg"]
			]
		)
		## bestTH = median
		X̂["WPT-BT_vs_med"], time["WPT-BT_vs_med"] = @timed hcat([
			denoise(
				Y["WPT-BT"][:,i], 
				:wpt, 
				wt, 
				tree=wpt_bt[:,i], 
				dnt=vs_dnt, 
				estnoise=median(σ₁["WPT-BT"])
			) for i in 1:samplesize
		]...)
		mean_psnr["WPT-BT_vs_med"] = mean(
			[psnr(X̂["WPT-BT_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT-BT_vs_med"] = mean(
			[ssim(X̂["WPT-BT_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-BT", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["WPT-BT_vs_med"], 
				mean_psnr["WPT-BT_vs_med"],
				mean_ssim["WPT-BT_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["WPT-BT_res_ind"], time["WPT-BT_res_ind"] = @timed hcat([
			denoise(
				Y["WPT-BT"][:,i], 
				:wpt, 
				wt, 
				tree=wpt_bt[:,i], 
				dnt=res_dnt, 
				estnoise=σ₂["WPT-BT"][i]
			) for i in 1:samplesize
		]...)
		mean_psnr["WPT-BT_res_ind"] = mean(
			[psnr(X̂["WPT-BT_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT-BT_res_ind"] = mean(
			[ssim(X̂["WPT-BT_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-BT", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["WPT-BT_res_ind"], 
				mean_psnr["WPT-BT_res_ind"],
				mean_ssim["WPT-BT_res_ind"]
			]
		)
		## bestTH = average
		X̂["WPT-BT_res_avg"], time["WPT-BT_res_avg"] = @timed hcat([
			denoise(
				Y["WPT-BT"][:,i], 
				:wpt, 
				wt, 
				tree=wpt_bt[:,i], 
				dnt=res_dnt, 
				estnoise=mean(σ₂["WPT-BT"])
			) for i in 1:samplesize
		]...)
		mean_psnr["WPT-BT_res_avg"] = mean(
			[psnr(X̂["WPT-BT_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT-BT_res_avg"] = mean(
			[ssim(X̂["WPT-BT_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-BT", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["WPT-BT_res_avg"], 
				mean_psnr["WPT-BT_res_avg"],
				mean_ssim["WPT-BT_res_avg"]
			]
		)
		## bestTH = median
		X̂["WPT-BT_res_med"], time["WPT-BT_res_med"] = @timed hcat([
			denoise(
				Y["WPT-BT"][:,i], 
				:wpt, 
				wt, 
				tree=wpt_bt[:,i], 
				dnt=res_dnt, 
				estnoise=median(σ₂["WPT-BT"])
			) for i in 1:samplesize
		]...)
		mean_psnr["WPT-BT_res_med"] = mean(
			[psnr(X̂["WPT-BT_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT-BT_res_med"] = mean(
			[ssim(X̂["WPT-BT_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-BT", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["WPT-BT_res_med"], 
				mean_psnr["WPT-BT_res_med"],
				mean_ssim["WPT-BT_res_med"]
			]
		)
	end
end;

# ╔═╡ 069497ac-fdae-4ddf-8983-026f7d46f07a
md"### 5. Joint Best Basis"

# ╔═╡ cd132003-1384-41a4-bfb4-91247630a24e
begin
	if autorun
		# Visushrink
		T["JBB"] = bestbasistree(Y["WPD"], JBB())
		Y["JBB"] = bestbasiscoef(Y["WPD"], T["JBB"])
		σ₁["JBB"] = [noisest(Y["JBB"][:,i], false, T["JBB"]) for i in axes(X,2)]
		σ₂["JBB"] = [relerrorthreshold(Y["JBB"][:,i], false, T["JBB"]) for i in axes(X,2)]
		## bestTH = individual
		X̂["JBB_vs_ind"], time["JBB_vs_ind"] = @timed denoiseall(
			Y["JBB"], 
			:wpt, 
			wt, 
			tree=T["JBB"], 
			estnoise=σ₁["JBB"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["JBB_vs_ind"] = mean(
			[psnr(X̂["JBB_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["JBB_vs_ind"] = mean(
			[ssim(X̂["JBB_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"JBB", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["JBB_vs_ind"], 
				mean_psnr["JBB_vs_ind"],
				mean_ssim["JBB_vs_ind"]
			]
		)
		## bestTH = average
		X̂["JBB_vs_avg"], time["JBB_vs_avg"] = @timed denoiseall(
			Y["JBB"], 
			:wpt, 
			wt, 
			tree=T["JBB"], 
			estnoise=σ₁["JBB"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["JBB_vs_avg"] = mean(
			[psnr(X̂["JBB_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["JBB_vs_avg"] = mean(
			[ssim(X̂["JBB_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"JBB", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["JBB_vs_avg"], 
				mean_psnr["JBB_vs_avg"],
				mean_ssim["JBB_vs_avg"]
			]
		)
		## bestTH = median
		X̂["JBB_vs_med"], time["JBB_vs_med"] = @timed denoiseall(
			Y["JBB"], 
			:wpt, 
			wt, 
			tree=T["JBB"], 
			estnoise=σ₁["JBB"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["JBB_vs_med"] = mean(
			[psnr(X̂["JBB_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["JBB_vs_med"] = mean(
			[ssim(X̂["JBB_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"JBB", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["JBB_vs_med"], 
				mean_psnr["JBB_vs_med"],
				mean_ssim["JBB_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["JBB_res_ind"], time["JBB_res_ind"] = @timed denoiseall(
			Y["JBB"], 
			:wpt, 
			wt, 
			tree=T["JBB"], 
			estnoise=σ₂["JBB"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["JBB_res_ind"] = mean(
			[psnr(X̂["JBB_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["JBB_res_ind"] = mean(
			[ssim(X̂["JBB_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"JBB", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["JBB_res_ind"], 
				mean_psnr["JBB_res_ind"],
				mean_ssim["JBB_res_ind"]
			]
		)
		## bestTH = average
		X̂["JBB_res_avg"], time["JBB_res_avg"] = @timed denoiseall(
			Y["JBB"], 
			:wpt, 
			wt, 
			tree=T["JBB"], 
			estnoise=σ₂["JBB"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["JBB_res_avg"] = mean(
			[psnr(X̂["JBB_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["JBB_res_avg"] = mean(
			[ssim(X̂["JBB_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"JBB", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["JBB_res_avg"], 
				mean_psnr["JBB_res_avg"],
				mean_ssim["JBB_res_avg"]
			]
		)
		## bestTH = median
		X̂["JBB_res_med"], time["JBB_res_med"] = @timed denoiseall(
			Y["JBB"], 
			:wpt, 
			wt, 
			tree=T["JBB"], 
			estnoise=σ₂["JBB"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["JBB_res_med"] = mean(
			[psnr(X̂["JBB_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["JBB_res_med"] = mean(
			[ssim(X̂["JBB_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"JBB", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["JBB_res_med"], 
				mean_psnr["JBB_res_med"],
				mean_ssim["JBB_res_med"]
			]
		)
	end
end;

# ╔═╡ b075d6d8-228a-4ce2-8647-e2c6b962ba48
md"### 6. Least Statistically Dependent Basis"

# ╔═╡ a7d6a82c-b143-4e8e-94ee-e8999eefc0f1
begin
	if autorun
		# Visushrink
		T["LSDB"] = bestbasistree(Y["WPD"], LSDB())
		Y["LSDB"] = bestbasiscoef(Y["WPD"], T["LSDB"])
		σ₁["LSDB"] = [noisest(Y["LSDB"][:,i], false, T["LSDB"]) for i in axes(X,2)]
		σ₂["LSDB"] = [relerrorthreshold(Y["LSDB"][:,i], false, T["LSDB"]) for i in axes(X,2)]
		## bestTH = individual
		X̂["LSDB_vs_ind"], time["LSDB_vs_ind"] = @timed denoiseall(
			Y["LSDB"], 
			:wpt, 
			wt, 
			tree=T["LSDB"], 
			estnoise=σ₁["LSDB"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["LSDB_vs_ind"] = mean(
			[psnr(X̂["LSDB_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["LSDB_vs_ind"] = mean(
			[ssim(X̂["LSDB_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"LSDB", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["LSDB_vs_ind"], 
				mean_psnr["LSDB_vs_ind"],
				mean_ssim["LSDB_vs_ind"]
			]
		)
		## bestTH = average
		X̂["LSDB_vs_avg"], time["LSDB_vs_avg"] = @timed denoiseall(
			Y["LSDB"], 
			:wpt, 
			wt, 
			tree=T["LSDB"], 
			estnoise=σ₁["LSDB"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["LSDB_vs_avg"] = mean(
			[psnr(X̂["LSDB_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["LSDB_vs_avg"] = mean(
			[ssim(X̂["LSDB_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"LSDB", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["LSDB_vs_avg"], 
				mean_psnr["LSDB_vs_avg"],
				mean_ssim["LSDB_vs_avg"]
			]
		)
		## bestTH = median
		X̂["LSDB_vs_med"], time["LSDB_vs_med"] = @timed denoiseall(
			Y["LSDB"], 
			:wpt, 
			wt, 
			tree=T["LSDB"], 
			estnoise=σ₁["LSDB"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["LSDB_vs_med"] = mean(
			[psnr(X̂["LSDB_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["LSDB_vs_med"] = mean(
			[ssim(X̂["LSDB_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"LSDB", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["LSDB_vs_med"], 
				mean_psnr["LSDB_vs_med"],
				mean_ssim["LSDB_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["LSDB_res_ind"], time["LSDB_res_ind"] = @timed denoiseall(
			Y["LSDB"], 
			:wpt, 
			wt, 
			tree=T["LSDB"], 
			estnoise=σ₂["LSDB"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["LSDB_res_ind"] = mean(
			[psnr(X̂["LSDB_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["LSDB_res_ind"] = mean(
			[ssim(X̂["LSDB_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"LSDB", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["LSDB_res_ind"], 
				mean_psnr["LSDB_res_ind"],
				mean_ssim["LSDB_res_ind"]
			]
		)
		## bestTH = average
		X̂["LSDB_res_avg"], time["LSDB_res_avg"] = @timed denoiseall(
			Y["LSDB"], 
			:wpt, 
			wt, 
			tree=T["LSDB"], 
			estnoise=σ₂["LSDB"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["LSDB_res_avg"] = mean(
			[psnr(X̂["LSDB_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["LSDB_res_avg"] = mean(
			[ssim(X̂["LSDB_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"LSDB", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["LSDB_res_avg"], 
				mean_psnr["LSDB_res_avg"],
				mean_ssim["LSDB_res_avg"]
			]
		)
		## bestTH = median
		X̂["LSDB_res_med"], time["LSDB_res_med"] = @timed denoiseall(
			Y["LSDB"], 
			:wpt, 
			wt, 
			tree=T["LSDB"], 
			estnoise=σ₂["LSDB"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["LSDB_res_med"] = mean(
			[psnr(X̂["LSDB_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["LSDB_res_med"] = mean(
			[ssim(X̂["LSDB_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"LSDB", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["LSDB_res_med"], 
				mean_psnr["LSDB_res_med"],
				mean_ssim["LSDB_res_med"]
			]
		)
	end
end;

# ╔═╡ f2f949f8-772f-4787-8883-0d96137f0924
md"# Redundant Transforms"

# ╔═╡ 3895472f-0a4f-4b7a-84f6-470208b5e8cc
md"## 1. Autocorrelation Wavelet Transforms"

# ╔═╡ 7c3ae1ea-887d-4af6-ba18-7fd06ea6354d
md"### 1.1 Autocorrelation Discrete Wavelet Transform"

# ╔═╡ 89a56a57-a5b9-4380-a618-97d8b901c01b
begin
	if autorun
		# Visushrink
		Y["ACWT"] = cat([acwt(X[:,i], wt) for i in axes(X,2)]..., dims=3)
		σ₁["ACWT"] = [noisest(Y["ACWT"][:,:,i], true) for i in axes(X,2)]
		σ₂["ACWT"] = [relerrorthreshold(Y["ACWT"][:,:,i], true) for i in axes(X,2)]
		## bestTH = individual
		X̂["ACWT_vs_ind"], time["ACWT_vs_ind"] = @timed denoiseall(
			Y["ACWT"], 
			:acwt, 
			wt, 
			L=max_dec_level, 
			estnoise=σ₁["ACWT"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["ACWT_vs_ind"] = mean(
			[psnr(X̂["ACWT_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWT_vs_ind"] = mean(
			[ssim(X̂["ACWT_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWT", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["ACWT_vs_ind"], 
				mean_psnr["ACWT_vs_ind"],
				mean_ssim["ACWT_vs_ind"]
			]
		)
		## bestTH = average
		X̂["ACWT_vs_avg"], time["ACWT_vs_avg"] = @timed denoiseall(
			Y["ACWT"], 
			:acwt, 
			wt, 
			L=max_dec_level, 
			estnoise=σ₁["ACWT"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["ACWT_vs_avg"] = mean(
			[psnr(X̂["ACWT_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWT_vs_avg"] = mean(
			[ssim(X̂["ACWT_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWT", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["ACWT_vs_avg"], 
				mean_psnr["ACWT_vs_avg"],
				mean_ssim["ACWT_vs_avg"]
			]
		)
		## bestTH = median
		X̂["ACWT_vs_med"], time["ACWT_vs_med"] = @timed denoiseall(
			Y["ACWT"], 
			:acwt, 
			wt, 
			L=max_dec_level, 
			estnoise=σ₁["ACWT"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["ACWT_vs_med"] = mean(
			[psnr(X̂["ACWT_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWT_vs_med"] = mean(
			[ssim(X̂["ACWT_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWT", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["ACWT_vs_med"], 
				mean_psnr["ACWT_vs_med"],
				mean_ssim["ACWT_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["ACWT_res_ind"], time["ACWT_res_ind"] = @timed denoiseall(
			Y["ACWT"], 
			:acwt, 
			wt, 
			L=max_dec_level, 
			estnoise=σ₂["ACWT"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["ACWT_res_ind"] = mean(
			[psnr(X̂["ACWT_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWT_res_ind"] = mean(
			[ssim(X̂["ACWT_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWT", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["ACWT_res_ind"], 
				mean_psnr["ACWT_res_ind"],
				mean_ssim["ACWT_res_ind"]
			]
		)
		## bestTH = average
		X̂["ACWT_res_avg"], time["ACWT_res_avg"] = @timed denoiseall(
			Y["ACWT"], 
			:acwt, 
			wt, 
			L=max_dec_level, 
			estnoise=σ₂["ACWT"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["ACWT_res_avg"] = mean(
			[psnr(X̂["ACWT_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWT_res_avg"] = mean(
			[ssim(X̂["ACWT_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWT", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["ACWT_res_avg"], 
				mean_psnr["ACWT_res_avg"],
				mean_ssim["ACWT_res_avg"]
			]
		)
		## bestTH = median
		X̂["ACWT_res_med"], time["ACWT_res_med"] = @timed denoiseall(
			Y["ACWT"], 
			:acwt, 
			wt, 
			L=max_dec_level, 
			estnoise=σ₂["ACWT"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["ACWT_res_med"] = mean(
			[psnr(X̂["ACWT_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWT_res_med"] = mean(
			[ssim(X̂["ACWT_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWT", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["ACWT_res_med"], 
				mean_psnr["ACWT_res_med"],
				mean_ssim["ACWT_res_med"]
			]
		)
	end
end;

# ╔═╡ 115edde2-b1ba-4d86-9b1a-e05d76026bcf
md"### 1.2 Autocorrelation Packet Transform - Level $(max_dec_level÷2)"

# ╔═╡ c52bd741-00cb-4cf2-97e3-b8dbba3af9ad
begin
	if autorun
		# Visushrink
		Y["ACWPT"] = cat(
			[acwpt(X[:,i], wt, max_dec_level) for i in axes(X,2)]..., dims=3
		)
		σ₁["ACWPT$(max_dec_level÷2)"] = [
			noisest(
				Y["ACWPT"][:,:,i], true, T["WPT$(max_dec_level÷2)"]
			) for i in axes(X,2)
		]
		σ₂["ACWPT$(max_dec_level÷2)"] = [
			relerrorthreshold(
				Y["ACWPT"][:,:,i], true, T["WPT$(max_dec_level÷2)"]
			) for i in axes(X,2)
		]
		## bestTH = individual
		X̂["ACWPT$(max_dec_level÷2)_vs_ind"], time["ACWPT$(max_dec_level÷2)_vs_ind"] =
		@timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT$(max_dec_level÷2)"], 
			estnoise=σ₁["ACWPT$(max_dec_level÷2)"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["ACWPT$(max_dec_level÷2)_vs_ind"] = mean(
			[
				psnr(
					X̂["ACWPT$(max_dec_level÷2)_vs_ind"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		mean_ssim["ACWPT$(max_dec_level÷2)_vs_ind"] = mean(
			[
				ssim(
					X̂["ACWPT$(max_dec_level÷2)_vs_ind"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		push!(
			results, 
			[
				"ACWPT-L$(max_dec_level÷2)", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["ACWPT$(max_dec_level÷2)_vs_ind"], 
				mean_psnr["ACWPT$(max_dec_level÷2)_vs_ind"],
				mean_ssim["ACWPT$(max_dec_level÷2)_vs_ind"]
			]
		)
		## bestTH = average
		X̂["ACWPT$(max_dec_level÷2)_vs_avg"], time["ACWPT$(max_dec_level÷2)_vs_avg"] =
		@timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT$(max_dec_level÷2)"], 
			estnoise=σ₁["ACWPT$(max_dec_level÷2)"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["ACWPT$(max_dec_level÷2)_vs_avg"] = mean(
			[
				psnr(
					X̂["ACWPT$(max_dec_level÷2)_vs_avg"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		mean_ssim["ACWPT$(max_dec_level÷2)_vs_avg"] = mean(
			[
				ssim(
					X̂["ACWPT$(max_dec_level÷2)_vs_avg"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		push!(
			results, 
			[
				"ACWPT-L$(max_dec_level÷2)", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["ACWPT$(max_dec_level÷2)_vs_avg"], 
				mean_psnr["ACWPT$(max_dec_level÷2)_vs_avg"],
				mean_ssim["ACWPT$(max_dec_level÷2)_vs_avg"]
			]
		)
		## bestTH = median
		X̂["ACWPT$(max_dec_level÷2)_vs_med"], time["ACWPT$(max_dec_level÷2)_vs_med"] =
		@timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT$(max_dec_level÷2)"], 
			estnoise=σ₁["ACWPT$(max_dec_level÷2)"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["ACWPT$(max_dec_level÷2)_vs_med"] = mean(
			[
				psnr(
					X̂["ACWPT$(max_dec_level÷2)_vs_med"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		mean_ssim["ACWPT$(max_dec_level÷2)_vs_med"] = mean(
			[
				ssim(
					X̂["ACWPT$(max_dec_level÷2)_vs_med"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		push!(
			results, 
			[
				"ACWPT-L$(max_dec_level÷2)", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["ACWPT$(max_dec_level÷2)_vs_med"], 
				mean_psnr["ACWPT$(max_dec_level÷2)_vs_med"],
				mean_ssim["ACWPT$(max_dec_level÷2)_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["ACWPT$(max_dec_level÷2)_res_ind"], time["ACWPT$(max_dec_level÷2)_res_ind"]=
		@timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT$(max_dec_level÷2)"], 
			estnoise=σ₂["ACWPT$(max_dec_level÷2)"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["ACWPT$(max_dec_level÷2)_res_ind"] = mean(
			[
				psnr(
					X̂["ACWPT$(max_dec_level÷2)_res_ind"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		mean_ssim["ACWPT$(max_dec_level÷2)_res_ind"] = mean(
			[
				ssim(
					X̂["ACWPT$(max_dec_level÷2)_res_ind"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		push!(
			results, 
			[
				"ACWPT-L$(max_dec_level÷2)", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["ACWPT$(max_dec_level÷2)_res_ind"], 
				mean_psnr["ACWPT$(max_dec_level÷2)_res_ind"],
				mean_ssim["ACWPT$(max_dec_level÷2)_res_ind"]
			]
		)
		## bestTH = average
		X̂["ACWPT$(max_dec_level÷2)_res_avg"], time["ACWPT$(max_dec_level÷2)_res_avg"]=
		@timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT$(max_dec_level÷2)"], 
			estnoise=σ₂["ACWPT$(max_dec_level÷2)"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["ACWPT$(max_dec_level÷2)_res_avg"] = mean(
			[
				psnr(
					X̂["ACWPT$(max_dec_level÷2)_res_avg"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		mean_ssim["ACWPT$(max_dec_level÷2)_res_avg"] = mean(
			[
				ssim(
					X̂["ACWPT$(max_dec_level÷2)_res_avg"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		push!(
			results, 
			[
				"ACWPT-L$(max_dec_level÷2)", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["ACWPT$(max_dec_level÷2)_res_avg"], 
				mean_psnr["ACWPT$(max_dec_level÷2)_res_avg"],
				mean_ssim["ACWPT$(max_dec_level÷2)_res_avg"]
			]
		)
		## bestTH = median
		X̂["ACWPT$(max_dec_level÷2)_res_med"], time["ACWPT$(max_dec_level÷2)_res_med"]=
		@timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT$(max_dec_level÷2)"], 
			estnoise=σ₂["ACWPT$(max_dec_level÷2)"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["ACWPT$(max_dec_level÷2)_res_med"] = mean(
			[
				psnr(
					X̂["ACWPT$(max_dec_level÷2)_res_med"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		mean_ssim["ACWPT$(max_dec_level÷2)_res_med"] = mean(
			[
				ssim(
					X̂["ACWPT$(max_dec_level÷2)_res_med"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		push!(
			results, 
			[
				"ACWPT-L$(max_dec_level÷2)", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["ACWPT$(max_dec_level÷2)_res_med"], 
				mean_psnr["ACWPT$(max_dec_level÷2)_res_med"],
				mean_ssim["ACWPT$(max_dec_level÷2)_res_med"]
			]
		)
	end
end;

# ╔═╡ a4b3694e-2ca5-46a5-b1ce-44c2f7ec2006
md"### 1.3 Autocorrelation Packet Transform - Level $(max_dec_level)"

# ╔═╡ 250d22dd-1d15-45d4-8aa4-3de1f37b164c
begin
	if autorun
		# Visushrink
		σ₁["ACWPT$(max_dec_level)"] = [
			noisest(
				Y["ACWPT"][:,:,i], true, T["WPT$(max_dec_level)"]
			) for i in axes(X,2)
		]
		σ₂["ACWPT$(max_dec_level)"] = [
			relerrorthreshold(
				Y["ACWPT"][:,:,i], true, T["WPT$(max_dec_level)"]
			) for i in axes(X,2)
		]
		## bestTH = individual
		X̂["ACWPT$(max_dec_level)_vs_ind"], time["ACWPT$(max_dec_level)_vs_ind"] =
		@timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT$(max_dec_level)"], 
			estnoise=σ₁["ACWPT$(max_dec_level)"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["ACWPT$(max_dec_level)_vs_ind"] = mean(
			[psnr(X̂["ACWPT$(max_dec_level)_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT$(max_dec_level)_vs_ind"] = mean(
			[ssim(X̂["ACWPT$(max_dec_level)_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-L$(max_dec_level)", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["ACWPT$(max_dec_level)_vs_ind"], 
				mean_psnr["ACWPT$(max_dec_level)_vs_ind"],
				mean_ssim["ACWPT$(max_dec_level)_vs_ind"]
			]
		)
		## bestTH = average
		X̂["ACWPT$(max_dec_level)_vs_avg"], time["ACWPT$(max_dec_level)_vs_avg"] =
		@timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT$(max_dec_level)"], 
			estnoise=σ₁["ACWPT$(max_dec_level)"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["ACWPT$(max_dec_level)_vs_avg"] = mean(
			[psnr(X̂["ACWPT$(max_dec_level)_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT$(max_dec_level)_vs_avg"] = mean(
			[ssim(X̂["ACWPT$(max_dec_level)_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-L$(max_dec_level)", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["ACWPT$(max_dec_level)_vs_avg"], 
				mean_psnr["ACWPT$(max_dec_level)_vs_avg"],
				mean_ssim["ACWPT$(max_dec_level)_vs_avg"]
			]
		)
		## bestTH = median
		X̂["ACWPT$(max_dec_level)_vs_med"], time["ACWPT$(max_dec_level)_vs_med"] = 
		@timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT$(max_dec_level)"], 
			estnoise=σ₁["ACWPT$(max_dec_level)"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["ACWPT$(max_dec_level)_vs_med"] = mean(
			[psnr(X̂["ACWPT$(max_dec_level)_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT$(max_dec_level)_vs_med"] = mean(
			[ssim(X̂["ACWPT$(max_dec_level)_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-L$(max_dec_level)", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["ACWPT$(max_dec_level)_vs_med"], 
				mean_psnr["ACWPT$(max_dec_level)_vs_med"],
				mean_ssim["ACWPT$(max_dec_level)_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["ACWPT$(max_dec_level)_res_ind"], time["ACWPT$(max_dec_level)_res_ind"] = 
		@timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT$(max_dec_level)"], 
			estnoise=σ₂["ACWPT$(max_dec_level)"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["ACWPT$(max_dec_level)_res_ind"] = mean(
			[psnr(X̂["ACWPT$(max_dec_level)_res_ind"][:,i],X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT$(max_dec_level)_res_ind"] = mean(
			[ssim(X̂["ACWPT$(max_dec_level)_res_ind"][:,i],X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-L$(max_dec_level)", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["ACWPT$(max_dec_level)_res_ind"], 
				mean_psnr["ACWPT$(max_dec_level)_res_ind"],
				mean_ssim["ACWPT$(max_dec_level)_res_ind"]
			]
		)
		## bestTH = average
		X̂["ACWPT$(max_dec_level)_res_avg"], time["ACWPT$(max_dec_level)_res_avg"] = 
		@timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT$(max_dec_level)"], 
			estnoise=σ₂["ACWPT$(max_dec_level)"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["ACWPT$(max_dec_level)_res_avg"] = mean(
			[psnr(X̂["ACWPT$(max_dec_level)_res_avg"][:,i],X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT$(max_dec_level)_res_avg"] = mean(
			[ssim(X̂["ACWPT$(max_dec_level)_res_avg"][:,i],X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-L$(max_dec_level)", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["ACWPT$(max_dec_level)_res_avg"], 
				mean_psnr["ACWPT$(max_dec_level)_res_avg"],
				mean_ssim["ACWPT$(max_dec_level)_res_avg"]
			]
		)
		## bestTH = median
		X̂["ACWPT$(max_dec_level)_res_med"], time["ACWPT$(max_dec_level)_res_med"] = 
		@timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT$(max_dec_level)"], 
			estnoise=σ₂["ACWPT$(max_dec_level)"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["ACWPT$(max_dec_level)_res_med"] = mean(
			[psnr(X̂["ACWPT$(max_dec_level)_res_med"][:,i],X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT$(max_dec_level)_res_med"] = mean(
			[ssim(X̂["ACWPT$(max_dec_level)_res_med"][:,i],X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-L$(max_dec_level)", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["ACWPT$(max_dec_level)_res_med"], 
				mean_psnr["ACWPT$(max_dec_level)_res_med"],
				mean_ssim["ACWPT$(max_dec_level)_res_med"]
			]
		)
	end
end;

# ╔═╡ 7d057183-c8b2-4ebd-9ee9-aa7998d9a6d5
md"### 1.4 Autocorrelation Packet Transform - Best Basis"

# ╔═╡ e484c5ca-1e21-4cc2-b1a6-7e1aa7958952
begin
	if autorun
		# Visushrink
		acwpt_bt = bestbasistree(Y["ACWPT"], BB(redundant=true))
		σ₁["ACWPT-BT"]= [
			noisest(Y["ACWPT"][:,:,i], true, acwpt_bt[:,i]) for i in axes(X,2)
		]
		σ₂["ACWPT-BT"]= [
			relerrorthreshold(
				Y["ACWPT"][:,:,i], true, acwpt_bt[:,i]
			) for i in axes(X,2)
		]
		## bestTH = individual
		X̂["ACWPT-BT_vs_ind"], time["ACWPT-BT_vs_ind"] = @timed hcat([
			denoise(
				Y["ACWPT"][:,:,i], 
				:acwpt, 
				wt, 
				tree=acwpt_bt[:,i], 
				dnt=vs_dnt, 
				estnoise=σ₁["ACWPT-BT"][i]
			) for i in 1:samplesize
		]...)
		mean_psnr["ACWPT-BT_vs_ind"] = mean(
			[psnr(X̂["ACWPT-BT_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT-BT_vs_ind"] = mean(
			[ssim(X̂["ACWPT-BT_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-BT", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["ACWPT-BT_vs_ind"], 
				mean_psnr["ACWPT-BT_vs_ind"],
				mean_ssim["ACWPT-BT_vs_ind"]
			]
		)
		## bestTH = average
		X̂["ACWPT-BT_vs_avg"], time["ACWPT-BT_vs_avg"] = @timed hcat([
			denoise(
				Y["ACWPT"][:,:,i], 
				:acwpt, 
				wt, 
				tree=acwpt_bt[:,i], 
				dnt=vs_dnt, 
				estnoise=mean(σ₁["ACWPT-BT"])
			) for i in 1:samplesize
		]...)
		mean_psnr["ACWPT-BT_vs_avg"] = mean(
			[psnr(X̂["ACWPT-BT_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT-BT_vs_avg"] = mean(
			[ssim(X̂["ACWPT-BT_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-BT", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["ACWPT-BT_vs_avg"], 
				mean_psnr["ACWPT-BT_vs_avg"],
				mean_ssim["ACWPT-BT_vs_avg"]
			]
		)
		## bestTH = median
		X̂["ACWPT-BT_vs_med"], time["ACWPT-BT_vs_med"] = @timed hcat([
			denoise(
				Y["ACWPT"][:,:,i], 
				:acwpt, 
				wt, 
				tree=acwpt_bt[:,i], 
				dnt=vs_dnt, 
				estnoise=median(σ₁["ACWPT-BT"])
			) for i in 1:samplesize
		]...)
		mean_psnr["ACWPT-BT_vs_med"] = mean(
			[psnr(X̂["ACWPT-BT_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT-BT_vs_med"] = mean(
			[ssim(X̂["ACWPT-BT_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-BT", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["ACWPT-BT_vs_med"], 
				mean_psnr["ACWPT-BT_vs_med"],
				mean_ssim["ACWPT-BT_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["ACWPT-BT_res_ind"], time["ACWPT-BT_res_ind"] = @timed hcat([
			denoise(
				Y["ACWPT"][:,:,i], 
				:acwpt, 
				wt, 
				tree=acwpt_bt[:,i], 
				dnt=res_dnt, 
				estnoise=σ₂["ACWPT-BT"][i]
			) for i in 1:samplesize
		]...)
		mean_psnr["ACWPT-BT_res_ind"] = mean(
			[psnr(X̂["ACWPT-BT_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT-BT_res_ind"] = mean(
			[ssim(X̂["ACWPT-BT_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-BT", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["ACWPT-BT_res_ind"], 
				mean_psnr["ACWPT-BT_res_ind"],
				mean_ssim["ACWPT-BT_res_ind"]
			]
		)
		## bestTH = average
		X̂["ACWPT-BT_res_avg"], time["ACWPT-BT_res_avg"] = @timed hcat([
			denoise(
				Y["ACWPT"][:,:,i], 
				:acwpt, 
				wt, 
				tree=acwpt_bt[:,i], 
				dnt=res_dnt, 
				estnoise=mean(σ₂["ACWPT-BT"])
			) for i in 1:samplesize
		]...)
		mean_psnr["ACWPT-BT_res_avg"] = mean(
			[psnr(X̂["ACWPT-BT_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT-BT_res_avg"] = mean(
			[ssim(X̂["ACWPT-BT_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-BT", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["ACWPT-BT_res_avg"], 
				mean_psnr["ACWPT-BT_res_avg"],
				mean_ssim["ACWPT-BT_res_avg"]
			]
		)
		## bestTH = median
		X̂["ACWPT-BT_res_med"], time["ACWPT-BT_res_med"] = @timed hcat([
			denoise(
				Y["ACWPT"][:,:,i], 
				:acwpt, 
				wt, 
				tree=acwpt_bt[:,i], 
				dnt=res_dnt, 
				estnoise=median(σ₂["ACWPT-BT"])
			) for i in 1:samplesize
		]...)
		mean_psnr["ACWPT-BT_res_med"] = mean(
			[psnr(X̂["ACWPT-BT_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT-BT_res_med"] = mean(
			[ssim(X̂["ACWPT-BT_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-BT", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["ACWPT-BT_res_med"], 
				mean_psnr["ACWPT-BT_res_med"],
				mean_ssim["ACWPT-BT_res_med"]
			]
		)
	end
end;

# ╔═╡ 084decfb-5c6f-466d-a5f8-ddf8cc863d8c
md"### 1.5 Autocorrelation Joint Best Basis"

# ╔═╡ c5a90584-fc46-4f7e-8633-6866001dadf6
begin
	if autorun
		# Visushrink
		T["ACJBB"] = bestbasistree(Y["ACWPT"], JBB(redundant=true))
		σ₁["ACJBB"] = [
			noisest(Y["ACWPT"][:,:,i], true, T["ACJBB"]) for i in axes(X,2)
		]
		σ₂["ACJBB"] = [
			relerrorthreshold(Y["ACWPT"][:,:,i], true, T["ACJBB"]) for i in axes(X,2)
		]
		## bestTH = individual
		X̂["ACJBB_vs_ind"], time["ACJBB_vs_ind"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["ACJBB"], 
			estnoise=σ₁["ACJBB"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["ACJBB_vs_ind"] = mean(
			[psnr(X̂["ACJBB_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACJBB_vs_ind"] = mean(
			[ssim(X̂["ACJBB_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACJBB", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["ACJBB_vs_ind"], 
				mean_psnr["ACJBB_vs_ind"],
				mean_ssim["ACJBB_vs_ind"]
			]
		)
		## bestTH = average
		X̂["ACJBB_vs_avg"], time["ACJBB_vs_avg"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["ACJBB"], 
			estnoise=σ₁["ACJBB"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["ACJBB_vs_avg"] = mean(
			[psnr(X̂["ACJBB_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACJBB_vs_avg"] = mean(
			[ssim(X̂["ACJBB_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACJBB", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["ACJBB_vs_avg"], 
				mean_psnr["ACJBB_vs_avg"],
				mean_ssim["ACJBB_vs_avg"]
			]
		)
		## bestTH = median
		X̂["ACJBB_vs_med"], time["ACJBB_vs_med"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["ACJBB"], 
			estnoise=σ₁["ACJBB"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["ACJBB_vs_med"] = mean(
			[psnr(X̂["ACJBB_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACJBB_vs_med"] = mean(
			[ssim(X̂["ACJBB_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACJBB", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["ACJBB_vs_med"], 
				mean_psnr["ACJBB_vs_med"],
				mean_ssim["ACJBB_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["ACJBB_res_ind"], time["ACJBB_res_ind"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["ACJBB"], 
			estnoise=σ₂["ACJBB"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["ACJBB_res_ind"] = mean(
			[psnr(X̂["ACJBB_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACJBB_res_ind"] = mean(
			[ssim(X̂["ACJBB_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACJBB", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["ACJBB_res_ind"], 
				mean_psnr["ACJBB_res_ind"],
				mean_ssim["ACJBB_res_ind"]
			]
		)
		## bestTH = average
		X̂["ACJBB_res_avg"], time["ACJBB_res_avg"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["ACJBB"], 
			estnoise=σ₂["ACJBB"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["ACJBB_res_avg"] = mean(
			[psnr(X̂["ACJBB_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACJBB_res_avg"] = mean(
			[ssim(X̂["ACJBB_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACJBB", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["ACJBB_res_avg"], 
				mean_psnr["ACJBB_res_avg"],
				mean_ssim["ACJBB_res_avg"]
			]
		)
		## bestTH = median
		X̂["ACJBB_res_med"], time["ACJBB_res_med"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["ACJBB"], 
			estnoise=σ₂["ACJBB"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["ACJBB_res_med"] = mean(
			[psnr(X̂["ACJBB_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACJBB_res_med"] = mean(
			[ssim(X̂["ACJBB_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACJBB", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["ACJBB_res_med"], 
				mean_psnr["ACJBB_res_med"],
				mean_ssim["ACJBB_res_med"]
			]
		)
	end
end;

# ╔═╡ 6059509f-2237-4309-88cc-e1e777e2abe0
md"### 1.6 Autocorrelation Least Statistically Dependent Basis"

# ╔═╡ 991f747c-c4ee-4a95-9745-85c728974af5
begin
	if autorun
		# Visushrink
		T["ACLSDB"] = bestbasistree(Y["ACWPT"], LSDB(redundant=true))
		σ₁["ACLSDB"] = [
			noisest(Y["ACWPT"][:,:,i], true, T["ACLSDB"]) for i in axes(X,2)
		]
		σ₂["ACLSDB"] = [
			relerrorthreshold(Y["ACWPT"][:,:,i], true, T["ACLSDB"]) for i in axes(X,2)
		]
		## bestTH = individual
		X̂["ACLSDB_vs_ind"], time["ACLSDB_vs_ind"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["ACLSDB"], 
			estnoise=σ₁["ACLSDB"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["ACLSDB_vs_ind"] = mean(
			[psnr(X̂["ACLSDB_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACLSDB_vs_ind"] = mean(
			[ssim(X̂["ACLSDB_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACLSDB", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["ACLSDB_vs_ind"], 
				mean_psnr["ACLSDB_vs_ind"],
				mean_ssim["ACLSDB_vs_ind"]
			]
		)
		## bestTH = average
		X̂["ACLSDB_vs_avg"], time["ACLSDB_vs_avg"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["ACLSDB"], 
			estnoise=σ₁["ACLSDB"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["ACLSDB_vs_avg"] = mean(
			[psnr(X̂["ACLSDB_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACLSDB_vs_avg"] = mean(
			[ssim(X̂["ACLSDB_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACLSDB", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["ACLSDB_vs_avg"], 
				mean_psnr["ACLSDB_vs_avg"],
				mean_ssim["ACLSDB_vs_avg"]
			]
		)
		## bestTH = median
		X̂["ACLSDB_vs_med"], time["ACLSDB_vs_med"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["ACLSDB"], 
			estnoise=σ₁["ACLSDB"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["ACLSDB_vs_med"] = mean(
			[psnr(X̂["ACLSDB_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACLSDB_vs_med"] = mean(
			[ssim(X̂["ACLSDB_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACLSDB", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["ACLSDB_vs_med"], 
				mean_psnr["ACLSDB_vs_med"],
				mean_ssim["ACLSDB_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["ACLSDB_res_ind"], time["ACLSDB_res_ind"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["ACLSDB"], 
			estnoise=σ₂["ACLSDB"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["ACLSDB_res_ind"] = mean(
			[psnr(X̂["ACLSDB_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACLSDB_res_ind"] = mean(
			[ssim(X̂["ACLSDB_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACLSDB", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["ACLSDB_res_ind"], 
				mean_psnr["ACLSDB_res_ind"],
				mean_ssim["ACLSDB_res_ind"]
			]
		)
		## bestTH = average
		X̂["ACLSDB_res_avg"], time["ACLSDB_res_avg"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["ACLSDB"], 
			estnoise=σ₂["ACLSDB"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["ACLSDB_res_avg"] = mean(
			[psnr(X̂["ACLSDB_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACLSDB_res_avg"] = mean(
			[ssim(X̂["ACLSDB_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACLSDB", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["ACLSDB_res_avg"], 
				mean_psnr["ACLSDB_res_avg"],
				mean_ssim["ACLSDB_res_avg"]
			]
		)
		## bestTH = median
		X̂["ACLSDB_res_med"], time["ACLSDB_res_med"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["ACLSDB"], 
			estnoise=σ₂["ACLSDB"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["ACLSDB_res_med"] = mean(
			[psnr(X̂["ACLSDB_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACLSDB_res_med"] = mean(
			[ssim(X̂["ACLSDB_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACLSDB", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["ACLSDB_res_med"], 
				mean_psnr["ACLSDB_res_med"],
				mean_ssim["ACLSDB_res_med"]
			]
		)
	end
end;

# ╔═╡ cb927f51-99f2-4eeb-a0e5-5f1c65464b6f
md"## 2. Stationary Wavelet Transforms"

# ╔═╡ d7da11fd-8768-4ac7-81af-82f68e794e1a
md"### 2.1 Stationary Discrete Wavelet Transform"

# ╔═╡ ab9b089f-fbb7-4436-8d6a-963db1e95670
begin
	if autorun
		# Visushrink
		Y["SDWT"] = cat([sdwt(X[:,i], wt) for i in axes(X,2)]..., dims=3)
		σ₁["SDWT"] = [noisest(Y["SDWT"][:,:,i], true) for i in axes(X,2)]
		σ₂["SDWT"] = [relerrorthreshold(Y["SDWT"][:,:,i], true) for i in axes(X,2)]
		## bestTH = individual
		X̂["SDWT_vs_ind"], time["SDWT_vs_ind"] = @timed denoiseall(
			Y["SDWT"], 
			:sdwt, 
			wt, 
			L=max_dec_level, 
			estnoise=σ₁["SDWT"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["SDWT_vs_ind"] = mean(
			[psnr(X̂["SDWT_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SDWT_vs_ind"] = mean(
			[ssim(X̂["SDWT_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SDWT", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["SDWT_vs_ind"], 
				mean_psnr["SDWT_vs_ind"],
				mean_ssim["SDWT_vs_ind"]
			]
		)
		## bestTH = average
		X̂["SDWT_vs_avg"], time["SDWT_vs_avg"] = @timed denoiseall(
			Y["SDWT"], 
			:sdwt, 
			wt, 
			L=max_dec_level, 
			estnoise=σ₁["SDWT"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["SDWT_vs_avg"] = mean(
			[psnr(X̂["SDWT_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SDWT_vs_avg"] = mean(
			[ssim(X̂["SDWT_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SDWT", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["SDWT_vs_avg"], 
				mean_psnr["SDWT_vs_avg"],
				mean_ssim["SDWT_vs_avg"]
			]
		)
		## bestTH = median
		X̂["SDWT_vs_med"], time["SDWT_vs_med"] = @timed denoiseall(
			Y["SDWT"], 
			:sdwt, 
			wt, 
			L=max_dec_level, 
			estnoise=σ₁["SDWT"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["SDWT_vs_med"] = mean(
			[psnr(X̂["SDWT_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SDWT_vs_med"] = mean(
			[ssim(X̂["SDWT_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SDWT", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["SDWT_vs_med"], 
				mean_psnr["SDWT_vs_med"],
				mean_ssim["SDWT_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["SDWT_res_ind"], time["SDWT_res_ind"] = @timed denoiseall(
			Y["SDWT"], 
			:sdwt, 
			wt, 
			L=max_dec_level, 
			estnoise=σ₂["SDWT"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["SDWT_res_ind"] = mean(
			[psnr(X̂["SDWT_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SDWT_res_ind"] = mean(
			[ssim(X̂["SDWT_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SDWT", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["SDWT_res_ind"], 
				mean_psnr["SDWT_res_ind"],
				mean_ssim["SDWT_res_ind"]
			]
		)
		## bestTH = average
		X̂["SDWT_res_avg"], time["SDWT_res_avg"] = @timed denoiseall(
			Y["SDWT"], 
			:sdwt, 
			wt, 
			L=max_dec_level, 
			estnoise=σ₂["SDWT"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["SDWT_res_avg"] = mean(
			[psnr(X̂["SDWT_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SDWT_res_avg"] = mean(
			[ssim(X̂["SDWT_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SDWT", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["SDWT_res_avg"], 
				mean_psnr["SDWT_res_avg"],
				mean_ssim["SDWT_res_avg"]
			]
		)
		## bestTH = median
		X̂["SDWT_res_med"], time["SDWT_res_med"] = @timed denoiseall(
			Y["SDWT"], 
			:sdwt, 
			wt, 
			L=max_dec_level, 
			estnoise=σ₂["SDWT"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["SDWT_res_med"] = mean(
			[psnr(X̂["SDWT_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SDWT_res_med"] = mean(
			[ssim(X̂["SDWT_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SDWT", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["SDWT_res_med"], 
				mean_psnr["SDWT_res_med"],
				mean_ssim["SDWT_res_med"]
			]
		)
	end
end;

# ╔═╡ 72204688-6346-4ff5-b443-839cbd7074d8
md"### 2.2 Stationary Wavelet Packet Transform - Level $(max_dec_level÷2)"

# ╔═╡ 6d09613a-5676-4b45-bf44-7e2c40bb71c9
begin
	if autorun
		# Visushrink
		Y["SWPD"] = cat(
			[swpd(X[:,i], wt, max_dec_level) for i in axes(X,2)]..., dims=3
		)
		σ₁["SWPT$(max_dec_level÷2)"] = [
			noisest(
				Y["SWPD"][:,:,i], true, T["WPT$(max_dec_level÷2)"]
			) for i in axes(X,2)
		]
		σ₂["SWPT$(max_dec_level÷2)"] = [
			relerrorthreshold(
				Y["SWPD"][:,:,i], true, T["WPT$(max_dec_level÷2)"]
			) for i in axes(X,2)
		]
		## bestTH = individual
		X̂["SWPT$(max_dec_level÷2)_vs_ind"], time["SWPT$(max_dec_level÷2)_vs_ind"] =
		@timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT$(max_dec_level÷2)"], 
			estnoise=σ₁["SWPT$(max_dec_level÷2)"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["SWPT$(max_dec_level÷2)_vs_ind"] = mean(
			[
				psnr(
					X̂["SWPT$(max_dec_level÷2)_vs_ind"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		mean_ssim["SWPT$(max_dec_level÷2)_vs_ind"] = mean(
			[
				ssim(
					X̂["SWPT$(max_dec_level÷2)_vs_ind"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		push!(
			results, 
			[
				"SWPT-L$(max_dec_level÷2)", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["SWPT$(max_dec_level÷2)_vs_ind"], 
				mean_psnr["SWPT$(max_dec_level÷2)_vs_ind"],
				mean_ssim["SWPT$(max_dec_level÷2)_vs_ind"]
			]
		)
		## bestTH = average
		X̂["SWPT$(max_dec_level÷2)_vs_avg"], time["SWPT$(max_dec_level÷2)_vs_avg"] =
		@timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT$(max_dec_level÷2)"], 
			estnoise=σ₁["SWPT$(max_dec_level÷2)"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["SWPT$(max_dec_level÷2)_vs_avg"] = mean(
			[
				psnr(
					X̂["SWPT$(max_dec_level÷2)_vs_avg"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		mean_ssim["SWPT$(max_dec_level÷2)_vs_avg"] = mean(
			[
				ssim(
					X̂["SWPT$(max_dec_level÷2)_vs_avg"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		push!(
			results, 
			[
				"SWPT-L$(max_dec_level÷2)", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["SWPT$(max_dec_level÷2)_vs_avg"], 
				mean_psnr["SWPT$(max_dec_level÷2)_vs_avg"],
				mean_ssim["SWPT$(max_dec_level÷2)_vs_avg"]
			]
		)
		## bestTH = median
		X̂["SWPT$(max_dec_level÷2)_vs_med"], time["SWPT$(max_dec_level÷2)_vs_med"] =
		@timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT$(max_dec_level÷2)"], 
			estnoise=σ₁["SWPT$(max_dec_level÷2)"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["SWPT$(max_dec_level÷2)_vs_med"] = mean(
			[
				psnr(
					X̂["SWPT$(max_dec_level÷2)_vs_med"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		mean_ssim["SWPT$(max_dec_level÷2)_vs_med"] = mean(
			[
				ssim(
					X̂["SWPT$(max_dec_level÷2)_vs_med"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		push!(
			results, 
			[
				"SWPT-L$(max_dec_level÷2)", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["SWPT$(max_dec_level÷2)_vs_med"], 
				mean_psnr["SWPT$(max_dec_level÷2)_vs_med"],
				mean_ssim["SWPT$(max_dec_level÷2)_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["SWPT$(max_dec_level÷2)_res_ind"], time["SWPT$(max_dec_level÷2)_res_ind"] =
		@timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT$(max_dec_level÷2)"], 
			estnoise=σ₂["SWPT$(max_dec_level÷2)"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["SWPT$(max_dec_level÷2)_res_ind"] = mean(
			[
				psnr(
					X̂["SWPT$(max_dec_level÷2)_res_ind"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		mean_ssim["SWPT$(max_dec_level÷2)_res_ind"] = mean(
			[
				ssim(
					X̂["SWPT$(max_dec_level÷2)_res_ind"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		push!(
			results, 
			[
				"SWPT-L$(max_dec_level÷2)", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["SWPT$(max_dec_level÷2)_res_ind"], 
				mean_psnr["SWPT$(max_dec_level÷2)_res_ind"],
				mean_ssim["SWPT$(max_dec_level÷2)_res_ind"]
			]
		)
		## bestTH = average
		X̂["SWPT$(max_dec_level÷2)_res_avg"], time["SWPT$(max_dec_level÷2)_res_avg"] =
		@timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT$(max_dec_level÷2)"], 
			estnoise=σ₂["SWPT$(max_dec_level÷2)"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["SWPT$(max_dec_level÷2)_res_avg"] = mean(
			[
				psnr(
					X̂["SWPT$(max_dec_level÷2)_res_avg"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		mean_ssim["SWPT$(max_dec_level÷2)_res_avg"] = mean(
			[
				ssim(
					X̂["SWPT$(max_dec_level÷2)_res_avg"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		push!(
			results, 
			[
				"SWPT-L$(max_dec_level÷2)", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["SWPT$(max_dec_level÷2)_res_avg"], 
				mean_psnr["SWPT$(max_dec_level÷2)_res_avg"],
				mean_ssim["SWPT$(max_dec_level÷2)_res_avg"]
			]
		)
		## bestTH = median
		X̂["SWPT$(max_dec_level÷2)_res_med"], time["SWPT$(max_dec_level÷2)_res_med"] =
		@timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT$(max_dec_level÷2)"], 
			estnoise=σ₂["SWPT$(max_dec_level÷2)"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["SWPT$(max_dec_level÷2)_res_med"] = mean(
			[
				psnr(
					X̂["SWPT$(max_dec_level÷2)_res_med"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		mean_ssim["SWPT$(max_dec_level÷2)_res_med"] = mean(
			[
				ssim(
					X̂["SWPT$(max_dec_level÷2)_res_med"][:,i], X₀[:,i]
				) for i in axes(X,2)
			]
		)
		push!(
			results, 
			[
				"SWPT-L$(max_dec_level÷2)", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["SWPT$(max_dec_level÷2)_res_med"], 
				mean_psnr["SWPT$(max_dec_level÷2)_res_med"],
				mean_ssim["SWPT$(max_dec_level÷2)_res_med"]
			]
		)
	end
end;

# ╔═╡ 25de9867-d737-496c-8e0c-29bcdca898e8
md"### 2.3 Stationary Wavelet Packet Transform - Level $(max_dec_level)"

# ╔═╡ 3525a716-0210-4cd2-b780-3f1c27297e09
begin
	if autorun
		# Visushrink
		σ₁["SWPT$(max_dec_level)"] = [
			noisest(
				Y["SWPD"][:,:,i], true, T["WPT$(max_dec_level)"]
			) for i in axes(X,2)
		]
		σ₂["SWPT$(max_dec_level)"] = [
			relerrorthreshold(
				Y["SWPD"][:,:,i], true, T["WPT$(max_dec_level)"]
			) for i in axes(X,2)
		]
		## bestTH = individual
		X̂["SWPT$(max_dec_level)_vs_ind"], time["SWPT$(max_dec_level)_vs_ind"] = 
		@timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT$(max_dec_level)"], 
			estnoise=σ₁["SWPT$(max_dec_level)"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["SWPT$(max_dec_level)_vs_ind"] = mean(
			[psnr(X̂["SWPT$(max_dec_level)_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT$(max_dec_level)_vs_ind"] = mean(
			[ssim(X̂["SWPT$(max_dec_level)_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-L$(max_dec_level)", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["SWPT$(max_dec_level)_vs_ind"], 
				mean_psnr["SWPT$(max_dec_level)_vs_ind"],
				mean_ssim["SWPT$(max_dec_level)_vs_ind"]
			]
		)
		## bestTH = average
		X̂["SWPT$(max_dec_level)_vs_avg"], time["SWPT$(max_dec_level)_vs_avg"] = 
		@timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT$(max_dec_level)"], 
			estnoise=σ₁["SWPT$(max_dec_level)"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["SWPT$(max_dec_level)_vs_avg"] = mean(
			[psnr(X̂["SWPT$(max_dec_level)_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT$(max_dec_level)_vs_avg"] = mean(
			[ssim(X̂["SWPT$(max_dec_level)_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-L$(max_dec_level)", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["SWPT$(max_dec_level)_vs_avg"], 
				mean_psnr["SWPT$(max_dec_level)_vs_avg"],
				mean_ssim["SWPT$(max_dec_level)_vs_avg"]
			]
		)
		## bestTH = median
		X̂["SWPT$(max_dec_level)_vs_med"], time["SWPT$(max_dec_level)_vs_med"] = 
		@timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT$(max_dec_level)"], 
			estnoise=σ₁["SWPT$(max_dec_level)"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["SWPT$(max_dec_level)_vs_med"] = mean(
			[psnr(X̂["SWPT$(max_dec_level)_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT$(max_dec_level)_vs_med"] = mean(
			[ssim(X̂["SWPT$(max_dec_level)_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-L$(max_dec_level)", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["SWPT$(max_dec_level)_vs_med"], 
				mean_psnr["SWPT$(max_dec_level)_vs_med"],
				mean_ssim["SWPT$(max_dec_level)_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["SWPT$(max_dec_level)_res_ind"], time["SWPT$(max_dec_level)_res_ind"] = 
		@timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT$(max_dec_level)"], 
			estnoise=σ₂["SWPT$(max_dec_level)"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["SWPT$(max_dec_level)_res_ind"] = mean(
			[psnr(X̂["SWPT$(max_dec_level)_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT$(max_dec_level)_res_ind"] = mean(
			[ssim(X̂["SWPT$(max_dec_level)_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-L$(max_dec_level)", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["SWPT$(max_dec_level)_res_ind"], 
				mean_psnr["SWPT$(max_dec_level)_res_ind"],
				mean_ssim["SWPT$(max_dec_level)_res_ind"]
			]
		)
		## bestTH = average
		X̂["SWPT$(max_dec_level)_res_avg"], time["SWPT$(max_dec_level)_res_avg"] =
		@timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT$(max_dec_level)"], 
			estnoise=σ₂["SWPT$(max_dec_level)"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["SWPT$(max_dec_level)_res_avg"] = mean(
			[psnr(X̂["SWPT$(max_dec_level)_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT$(max_dec_level)_res_avg"] = mean(
			[ssim(X̂["SWPT$(max_dec_level)_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-L$(max_dec_level)", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["SWPT$(max_dec_level)_res_avg"], 
				mean_psnr["SWPT$(max_dec_level)_res_avg"],
				mean_ssim["SWPT$(max_dec_level)_res_avg"]
			]
		)
		## bestTH = median
		X̂["SWPT$(max_dec_level)_res_med"], time["SWPT$(max_dec_level)_res_med"] =
		@timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT$(max_dec_level)"], 
			estnoise=σ₂["SWPT$(max_dec_level)"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["SWPT$(max_dec_level)_res_med"] = mean(
			[psnr(X̂["SWPT$(max_dec_level)_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT$(max_dec_level)_res_med"] = mean(
			[ssim(X̂["SWPT$(max_dec_level)_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-L$(max_dec_level)", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["SWPT$(max_dec_level)_res_med"], 
				mean_psnr["SWPT$(max_dec_level)_res_med"],
				mean_ssim["SWPT$(max_dec_level)_res_med"]
			]
		)
	end
end;

# ╔═╡ 8774496e-a184-4c9a-9335-d2f184673cf5
md"### 2.4 Stationary Wavelet Packet Transform - Best Basis"

# ╔═╡ 609937c6-8b9f-4987-8f46-ec55ce05e861
begin
	if autorun
		# Visushrink
		swpt_bt = bestbasistree(Y["SWPD"], BB(redundant=true))
		σ₁["SWPT-BT"]= [
			noisest(Y["SWPD"][:,:,i], true, swpt_bt[:,i]) for i in axes(X,2)
		]
		σ₂["SWPT-BT"]= [
			relerrorthreshold(
				Y["SWPD"][:,:,i], true, swpt_bt[:,i]
			) for i in axes(X,2)
		]
		## bestTH = individual
		X̂["SWPT-BT_vs_ind"], time["SWPT-BT_vs_ind"] = @timed hcat([
			denoise(
				Y["SWPD"][:,:,i], 
				:swpd, 
				wt, 
				tree=swpt_bt[:,i], 
				dnt=vs_dnt, 
				estnoise=σ₁["SWPT-BT"][i]
			) for i in 1:samplesize
		]...)
		mean_psnr["SWPT-BT_vs_ind"] = mean(
			[psnr(X̂["SWPT-BT_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT-BT_vs_ind"] = mean(
			[ssim(X̂["SWPT-BT_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-BT", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["SWPT-BT_vs_ind"], 
				mean_psnr["SWPT-BT_vs_ind"],
				mean_ssim["SWPT-BT_vs_ind"]
			]
		)
		## bestTH = average
		X̂["SWPT-BT_vs_avg"], time["SWPT-BT_vs_avg"] = @timed hcat([
			denoise(
				Y["SWPD"][:,:,i], 
				:swpd, 
				wt, 
				tree=swpt_bt[:,i], 
				dnt=vs_dnt, 
				estnoise=mean(σ₁["SWPT-BT"])
			) for i in 1:samplesize
		]...)
		mean_psnr["SWPT-BT_vs_avg"] = mean(
			[psnr(X̂["SWPT-BT_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT-BT_vs_avg"] = mean(
			[ssim(X̂["SWPT-BT_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-BT", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["SWPT-BT_vs_avg"], 
				mean_psnr["SWPT-BT_vs_avg"],
				mean_ssim["SWPT-BT_vs_avg"]
			]
		)
		## bestTH = median
		X̂["SWPT-BT_vs_med"], time["SWPT-BT_vs_med"] = @timed hcat([
			denoise(
				Y["SWPD"][:,:,i], 
				:swpd, 
				wt, 
				tree=swpt_bt[:,i], 
				dnt=vs_dnt, 
				estnoise=median(σ₁["SWPT-BT"])
			) for i in 1:samplesize
		]...)
		mean_psnr["SWPT-BT_vs_med"] = mean(
			[psnr(X̂["SWPT-BT_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT-BT_vs_med"] = mean(
			[ssim(X̂["SWPT-BT_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-BT", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["SWPT-BT_vs_med"], 
				mean_psnr["SWPT-BT_vs_med"],
				mean_ssim["SWPT-BT_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["SWPT-BT_res_ind"], time["SWPT-BT_res_ind"] = @timed hcat([
			denoise(
				Y["SWPD"][:,:,i], 
				:swpd, 
				wt, 
				tree=swpt_bt[:,i], 
				dnt=res_dnt, 
				estnoise=σ₂["SWPT-BT"][i]
			) for i in 1:samplesize
		]...)
		mean_psnr["SWPT-BT_res_ind"] = mean(
			[psnr(X̂["SWPT-BT_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT-BT_res_ind"] = mean(
			[ssim(X̂["SWPT-BT_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-BT", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["SWPT-BT_res_ind"], 
				mean_psnr["SWPT-BT_res_ind"],
				mean_ssim["SWPT-BT_res_ind"]
			]
		)
		## bestTH = average
		X̂["SWPT-BT_res_avg"], time["SWPT-BT_res_avg"] = @timed hcat([
			denoise(
				Y["SWPD"][:,:,i], 
				:swpd, 
				wt, 
				tree=swpt_bt[:,i], 
				dnt=res_dnt, 
				estnoise=mean(σ₂["SWPT-BT"])
			) for i in 1:samplesize
		]...)
		mean_psnr["SWPT-BT_res_avg"] = mean(
			[psnr(X̂["SWPT-BT_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT-BT_res_avg"] = mean(
			[ssim(X̂["SWPT-BT_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-BT", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["SWPT-BT_res_avg"], 
				mean_psnr["SWPT-BT_res_avg"],
				mean_ssim["SWPT-BT_res_avg"]
			]
		)
		## bestTH = median
		X̂["SWPT-BT_res_med"], time["SWPT-BT_res_med"] = @timed hcat([
			denoise(
				Y["SWPD"][:,:,i], 
				:swpd, 
				wt, 
				tree=swpt_bt[:,i], 
				dnt=res_dnt, 
				estnoise=median(σ₂["SWPT-BT"])
			) for i in 1:samplesize
		]...)
		mean_psnr["SWPT-BT_res_med"] = mean(
			[psnr(X̂["SWPT-BT_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT-BT_res_med"] = mean(
			[ssim(X̂["SWPT-BT_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-BT", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["SWPT-BT_res_med"], 
				mean_psnr["SWPT-BT_res_med"],
				mean_ssim["SWPT-BT_res_med"]
			]
		)
	end
end;

# ╔═╡ aa5fff0c-9182-4568-b657-ca48c33de141
md"### 2.5 Stationary Joint Best Basis"

# ╔═╡ 7a6c247e-d765-4299-bd93-8d8271ca711f
begin
	if autorun
		# Visushrink
		T["SJBB"] = bestbasistree(Y["SWPD"], JBB(redundant=true))
		σ₁["SJBB"] = [noisest(Y["SWPD"][:,:,i], true, T["SJBB"]) for i in axes(X,2)]
		σ₂["SJBB"] = [relerrorthreshold(Y["SWPD"][:,:,i], true, T["SJBB"]) for i in axes(X,2)]
		## bestTH = individual
		X̂["SJBB_vs_ind"], time["SJBB_vs_ind"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["SJBB"], 
			estnoise=σ₁["SJBB"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["SJBB_vs_ind"] = mean(
			[psnr(X̂["SJBB_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SJBB_vs_ind"] = mean(
			[ssim(X̂["SJBB_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SJBB", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["SJBB_vs_ind"], 
				mean_psnr["SJBB_vs_ind"],
				mean_ssim["SJBB_vs_ind"]
			]
		)
		## bestTH = average
		X̂["SJBB_vs_avg"], time["SJBB_vs_avg"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["SJBB"], 
			estnoise=σ₁["SJBB"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["SJBB_vs_avg"] = mean(
			[psnr(X̂["SJBB_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SJBB_vs_avg"] = mean(
			[ssim(X̂["SJBB_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SJBB", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["SJBB_vs_avg"], 
				mean_psnr["SJBB_vs_avg"],
				mean_ssim["SJBB_vs_avg"]
			]
		)
		## bestTH = median
		X̂["SJBB_vs_med"], time["SJBB_vs_med"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["SJBB"], 
			estnoise=σ₁["SJBB"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["SJBB_vs_med"] = mean(
			[psnr(X̂["SJBB_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SJBB_vs_med"] = mean(
			[ssim(X̂["SJBB_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SJBB", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["SJBB_vs_med"], 
				mean_psnr["SJBB_vs_med"],
				mean_ssim["SJBB_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["SJBB_res_ind"], time["SJBB_res_ind"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["SJBB"], 
			estnoise=σ₂["SJBB"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["SJBB_res_ind"] = mean(
			[psnr(X̂["SJBB_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SJBB_res_ind"] = mean(
			[ssim(X̂["SJBB_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SJBB", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["SJBB_res_ind"], 
				mean_psnr["SJBB_res_ind"],
				mean_ssim["SJBB_res_ind"]
			]
		)
		## bestTH = average
		X̂["SJBB_res_avg"], time["SJBB_res_avg"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["SJBB"], 
			estnoise=σ₂["SJBB"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["SJBB_res_avg"] = mean(
			[psnr(X̂["SJBB_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SJBB_res_avg"] = mean(
			[ssim(X̂["SJBB_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SJBB", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["SJBB_res_avg"], 
				mean_psnr["SJBB_res_avg"],
				mean_ssim["SJBB_res_avg"]
			]
		)
		## bestTH = median
		X̂["SJBB_res_med"], time["SJBB_res_med"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["SJBB"], 
			estnoise=σ₂["SJBB"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["SJBB_res_med"] = mean(
			[psnr(X̂["SJBB_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SJBB_res_med"] = mean(
			[ssim(X̂["SJBB_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SJBB", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["SJBB_res_med"], 
				mean_psnr["SJBB_res_med"],
				mean_ssim["SJBB_res_med"]
			]
		)
	end
end;

# ╔═╡ ad509647-538a-446c-ae96-ca3dc8900ed5
md"### 2.6 Stationary Least Statistically Dependent Basis"

# ╔═╡ 94fb81cf-d565-456b-8687-3e9545330476
begin
	if autorun
		# Visushrink
		T["SLSDB"] = bestbasistree(Y["SWPD"], LSDB(redundant=true))
		σ₁["SLSDB"] = [noisest(Y["SWPD"][:,:,i], true, T["SLSDB"]) for i in axes(X,2)]
		σ₂["SLSDB"] = [relerrorthreshold(Y["SWPD"][:,:,i], true, T["SLSDB"]) for i in axes(X,2)]
		## bestTH = individual
		X̂["SLSDB_vs_ind"], time["SLSDB_vs_ind"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["SLSDB"], 
			estnoise=σ₁["SLSDB"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["SLSDB_vs_ind"] = mean(
			[psnr(X̂["SLSDB_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SLSDB_vs_ind"] = mean(
			[ssim(X̂["SLSDB_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SLSDB", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["SLSDB_vs_ind"], 
				mean_psnr["SLSDB_vs_ind"],
				mean_ssim["SLSDB_vs_ind"]
			]
		)
		## bestTH = average
		X̂["SLSDB_vs_avg"], time["SLSDB_vs_avg"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["SLSDB"], 
			estnoise=σ₁["SLSDB"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["SLSDB_vs_avg"] = mean(
			[psnr(X̂["SLSDB_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SLSDB_vs_avg"] = mean(
			[ssim(X̂["SLSDB_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SLSDB", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["SLSDB_vs_avg"],
				mean_psnr["SLSDB_vs_avg"],
				mean_ssim["SLSDB_vs_avg"]
			]
		)
		## bestTH = median
		X̂["SLSDB_vs_med"], time["SLSDB_vs_med"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["SLSDB"], 
			estnoise=σ₁["SLSDB"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["SLSDB_vs_med"] = mean(
			[psnr(X̂["SLSDB_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SLSDB_vs_med"] = mean(
			[ssim(X̂["SLSDB_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SLSDB", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["SLSDB_vs_med"], 
				mean_psnr["SLSDB_vs_med"],
				mean_ssim["SLSDB_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["SLSDB_res_ind"], time["SLSDB_res_ind"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["SLSDB"], 
			estnoise=σ₂["SLSDB"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["SLSDB_res_ind"] = mean(
			[psnr(X̂["SLSDB_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SLSDB_res_ind"] = mean(
			[ssim(X̂["SLSDB_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SLSDB", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["SLSDB_res_ind"], 
				mean_psnr["SLSDB_res_ind"],
				mean_ssim["SLSDB_res_ind"]
			]
		)
		## bestTH = average
		X̂["SLSDB_res_avg"], time["SLSDB_res_avg"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["SLSDB"], 
			estnoise=σ₂["SLSDB"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["SLSDB_res_avg"] = mean(
			[psnr(X̂["SLSDB_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SLSDB_res_avg"] = mean(
			[ssim(X̂["SLSDB_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SLSDB", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["SLSDB_res_avg"], 
				mean_psnr["SLSDB_res_avg"],
				mean_ssim["SLSDB_res_avg"]
			]
		)
		## bestTH = median
		X̂["SLSDB_res_med"], time["SLSDB_res_med"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["SLSDB"], 
			estnoise=σ₂["SLSDB"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["SLSDB_res_med"] = mean(
			[psnr(X̂["SLSDB_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SLSDB_res_med"] = mean(
			[ssim(X̂["SLSDB_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SLSDB", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["SLSDB_res_med"], 
				mean_psnr["SLSDB_res_med"],
				mean_ssim["SLSDB_res_med"]
			]
		)
	end
end;

# ╔═╡ 7103af68-9ad7-4b81-8c21-c8b6d7c9f5be
md"# Results Analysis"

# ╔═╡ 1682b2eb-b68c-4182-986b-d921fcfc039d
md"""
**Result Summary**  

To refresh and reorder the data frame below, you'll need to uncheck the following checkbox, make the necessary changes, and check the checkbox again.

Show Results $(@bind show_results CheckBox())

Number of results shown:
$(@bind topN Slider(1:109, default=5, show_value=true))
"""

# ╔═╡ 2aa29e02-6ff5-4226-9be3-367abbdebf22
md"""
Sort by:
$(@bind sortby Select([":PSNR" => "PSNR", ":SSIM" => "SSIM", ":time" => "Time"], default=":PSNR"))
"""

# ╔═╡ a392aef5-0df4-477c-a40e-48f206ee1c8d
md"""
Order:
$(@bind sortorder Select(["true" => "Decreasing", "false" => "Increasing"], 
	default = "true"))
"""

# ╔═╡ 0944c544-006a-45cc-b4c9-ad5bf6877ca3
begin
	if show_results
		column = eval(Meta.parse(sortby))
		ascending = eval(Meta.parse(sortorder))
		first(sort(results, [column], rev=ascending), topN)
	else
		nothing
	end
end

# ╔═╡ 9818ca36-0acc-4cc3-924e-8b50813c1da1
begin
	if show_results
		md"""
		**Selected Test Parameters:**

		* Test signal: $signal_name

		* Noise magnitude: $noise_size

		* Wavelet type: $wavelet_type

		* Threshold method: $threshold_method

		* Sample size: $samplesize

		**Note**: You might need to re-run the block below after updating a parameter to display the results.
		"""
	end
end

# ╔═╡ 56c7a1b9-c75f-48d8-a602-c219a2f432af
if show_results
	set_default_plot_size(6inch, 7inch)
	Gadfly.plot(
		results[results[:,:transform] .!= "None", :],
		y=:transform, 
		x=:PSNR,
		ygroup=:shrinking,
		color=:selection,
		xintercept=results[results[:, :transform] .== "None", :PSNR],
		Guide.title("PSNR"),
		Geom.subplot_grid(
            Geom.point, Geom.vline(color="red"), free_y_axis=true
        )
	)
end

# ╔═╡ 174edf71-3922-4f33-9024-caaaff884889
if show_results
	set_default_plot_size(6inch, 7inch)
	Gadfly.plot(
		results[results[:,:transform] .!= "None", :],
		y=:transform, 
		x=:SSIM,
		ygroup=:shrinking,
		color=:selection,
		xintercept=results[results[:, :transform] .== "None", :SSIM],
		Guide.title("SSIM"),
		Geom.subplot_grid(
            Geom.point, Geom.vline(color="red"), free_y_axis=true
        )
	)
end

# ╔═╡ Cell order:
# ╟─53c257e0-96ba-11eb-3615-8bfed63b2c18
# ╟─78225642-d1bb-41ca-9b77-1b9ad1b5d9a0
# ╟─ce4bf94e-edef-40c2-8ac5-b741e47a1759
# ╠═f9e001ae-0323-4283-9af1-1a8252b503e7
# ╟─85c26ead-f043-42c8-8245-58c8d03a963d
# ╠═203fb8c8-4358-4908-b616-a691ce329c02
# ╟─46e30602-a850-4175-b4bf-b4ef4b5359aa
# ╟─b0a28618-7cda-4c05-83d2-b54bbca3f9b5
# ╟─bb147649-36bf-4a82-95bf-2d5080873028
# ╟─ff1bfee3-2f30-4d15-9f3a-e3f422e67d72
# ╟─458a5a1e-c453-4199-befe-2bf4db6825ae
# ╟─7c0dba17-baf3-4b9c-b1c5-486f7e4515f4
# ╠═a184ae65-7947-4ffb-b751-8b6e97a8608b
# ╟─faa0a4ea-7849-408a-ac0f-c5cca8761aee
# ╟─851b04bb-e382-4a0a-98f6-7d4b983ca5ab
# ╟─356d75f4-6cc1-4062-8ef6-3cc6a9c2d9a7
# ╟─341c2551-b625-47a0-9163-3c0c0e7d4e13
# ╟─9eae2f5c-1f47-43d8-a7ec-0767e50e6a9b
# ╟─dbed8579-afa4-4a4d-b0bb-bd34877fa272
# ╟─e214b298-04d8-473f-b56e-5f446374078c
# ╟─e49e76e6-7018-4f49-a189-d2fae7df956d
# ╟─9b4ef541-9a36-4bc0-8654-10ab0a4e63b3
# ╟─4669be94-6c4c-42e2-b9d9-2dc98f1bdaea
# ╟─11e63c9a-6124-4122-9a86-ceed926d25d2
# ╟─d881753b-0432-451b-8de0-38a0b4b4382a
# ╟─8055194b-2e46-4d18-81c0-0c52bc3eb233
# ╟─f6277c19-1989-449e-96ba-6f81db68c76b
# ╟─56ee2c61-d83c-4d76-890a-a9bd0d65cee5
# ╟─c50ac92e-3684-4d0a-a80d-4ee9d74ec992
# ╟─e0a96592-5e77-4c29-9744-31369eea8147
# ╟─c178527f-96a4-4ac7-bb0c-38b73b38c45b
# ╟─ef3e7b66-fba0-467a-8a73-c9bf31fadbe3
# ╟─cd9e259e-8bb3-497b-ac7f-f89a003c8032
# ╟─3246e8b5-251f-4398-b21c-397341f2542e
# ╟─82e713f8-c870-43d2-a849-e3b401b00459
# ╟─6b02c425-39b9-467f-9406-3e9096873af4
# ╟─95081e88-a623-4e91-99c1-8b254b366dac
# ╟─126c41e7-dd65-46c6-8c5b-2439f5624fd5
# ╟─17bdc97a-4a0b-4931-a5c6-866f0c814601
# ╟─01e43234-2194-451d-9010-176aa4799fdb
# ╟─ae8059bd-5b5b-4ff2-a6f0-5ce672bdd54d
# ╟─cf55c5cb-ead6-40b6-896a-8f7e01613a46
# ╟─c95ebbed-3d9a-4be2-943b-08c86923ad89
# ╟─61d745d8-5c74-479b-9698-cd50bb68b3c7
# ╟─e5738623-4ea8-4866-a1da-7e849960f4e0
# ╟─c34c1f0c-7cfd-404b-aa59-d4bb6aa9628f
# ╟─069497ac-fdae-4ddf-8983-026f7d46f07a
# ╟─cd132003-1384-41a4-bfb4-91247630a24e
# ╟─b075d6d8-228a-4ce2-8647-e2c6b962ba48
# ╟─a7d6a82c-b143-4e8e-94ee-e8999eefc0f1
# ╟─f2f949f8-772f-4787-8883-0d96137f0924
# ╟─3895472f-0a4f-4b7a-84f6-470208b5e8cc
# ╟─7c3ae1ea-887d-4af6-ba18-7fd06ea6354d
# ╟─89a56a57-a5b9-4380-a618-97d8b901c01b
# ╟─115edde2-b1ba-4d86-9b1a-e05d76026bcf
# ╟─c52bd741-00cb-4cf2-97e3-b8dbba3af9ad
# ╟─a4b3694e-2ca5-46a5-b1ce-44c2f7ec2006
# ╟─250d22dd-1d15-45d4-8aa4-3de1f37b164c
# ╟─7d057183-c8b2-4ebd-9ee9-aa7998d9a6d5
# ╟─e484c5ca-1e21-4cc2-b1a6-7e1aa7958952
# ╟─084decfb-5c6f-466d-a5f8-ddf8cc863d8c
# ╟─c5a90584-fc46-4f7e-8633-6866001dadf6
# ╟─6059509f-2237-4309-88cc-e1e777e2abe0
# ╟─991f747c-c4ee-4a95-9745-85c728974af5
# ╟─cb927f51-99f2-4eeb-a0e5-5f1c65464b6f
# ╟─d7da11fd-8768-4ac7-81af-82f68e794e1a
# ╟─ab9b089f-fbb7-4436-8d6a-963db1e95670
# ╟─72204688-6346-4ff5-b443-839cbd7074d8
# ╟─6d09613a-5676-4b45-bf44-7e2c40bb71c9
# ╟─25de9867-d737-496c-8e0c-29bcdca898e8
# ╟─3525a716-0210-4cd2-b780-3f1c27297e09
# ╟─8774496e-a184-4c9a-9335-d2f184673cf5
# ╟─609937c6-8b9f-4987-8f46-ec55ce05e861
# ╟─aa5fff0c-9182-4568-b657-ca48c33de141
# ╟─7a6c247e-d765-4299-bd93-8d8271ca711f
# ╟─ad509647-538a-446c-ae96-ca3dc8900ed5
# ╟─94fb81cf-d565-456b-8687-3e9545330476
# ╟─7103af68-9ad7-4b81-8c21-c8b6d7c9f5be
# ╟─1682b2eb-b68c-4182-986b-d921fcfc039d
# ╟─2aa29e02-6ff5-4226-9be3-367abbdebf22
# ╟─a392aef5-0df4-477c-a40e-48f206ee1c8d
# ╟─0944c544-006a-45cc-b4c9-ad5bf6877ca3
# ╟─9818ca36-0acc-4cc3-924e-8b50813c1da1
# ╟─56c7a1b9-c75f-48d8-a602-c219a2f432af
# ╟─174edf71-3922-4f33-9024-caaaff884889
