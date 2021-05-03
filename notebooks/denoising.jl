### A Pluto.jl notebook ###
# v0.14.4

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
using Wavelets, LinearAlgebra, Plots, DataFrames, CSV, PlutoUI, WaveletsExt, Statistics, Gadfly

# ╔═╡ d0c98a14-ad3e-4a0b-b889-a3ea86f888f3
begin
	try     # open from local directory
		global testdata = CSV.read("../data/wavelet_test_256.csv", DataFrame)
	catch   # open from URL
		using HTTP
		url = HTTP.get("https://raw.githubusercontent.com/zengfung/WaveletsDenoisingExperiment/master/data/wavelet_test_256.csv")
		global testdata = CSV.read(url.body, DataFrame)
	end
end

# ╔═╡ 53c257e0-96ba-11eb-3615-8bfed63b2c18
md"# Denoising Experiment"

# ╔═╡ ce4bf94e-edef-40c2-8ac5-b741e47a1759
md"### Activate environment"

# ╔═╡ 85c26ead-f043-42c8-8245-58c8d03a963d
md"### Import libraries"

# ╔═╡ 46e30602-a850-4175-b4bf-b4ef4b5359aa
md"## I. Exploratory Data Analysis"

# ╔═╡ bc641876-9723-4c68-8221-ad03fb695c82
md"Let's load some test functions"

# ╔═╡ b0a28618-7cda-4c05-83d2-b54bbca3f9b5
md"**Select** a test function"

# ╔═╡ 7364da28-6a01-4359-9664-a3097e8bf1f1
@bind signal_name_test Select(
	["blocks", "bumps", "heavy_sine", "doppler", "quadchirp", "mishmash"],
	default = "doppler"
)

# ╔═╡ 458a5a1e-c453-4199-befe-2bf4db6825ae
md"**Adjust** the magnitude of Gaussian noise"

# ╔═╡ 97f40df0-9ccc-4e41-bebf-4e7188f33fff
@bind noise_size_test Slider(0:0.01:1, default=0.3, show_value=true)

# ╔═╡ 7c0dba17-baf3-4b9c-b1c5-486f7e4515f4
md"The `addnoise` function will add Gaussian noise that is proportional to the total energy of the signal."

# ╔═╡ a184ae65-7947-4ffb-b751-8b6e97a8608b
function addnoise(x::AbstractArray{<:Number,1}, s::Real=0.1)
	ϵ = randn(length(x))
	ϵ = ϵ/norm(ϵ)
	y = x + ϵ * s * norm(x)
	return y
end;

# ╔═╡ 851b04bb-e382-4a0a-98f6-7d4b983ca5ab
begin
	x_test = testdata[!,signal_name_test]
	p1_test = Plots.plot(x_test, ylim = (minimum(x_test)-1,maximum(x_test)+1), label = "Original signal", title = "Original vs Noisy")
	x_noisy_test = addnoise(x_test, noise_size_test)
	Plots.plot!(x_noisy_test, label = "Noisy signal");
	
	y_test = acwt(x_noisy_test, wavelet(WT.db4), 4);
	p2_test = WaveletsExt.wiggle(y_test)
	Plots.plot!(p2_test, title = "Autocorrelation Transform of Noisy Signal")
	
	z_test = sdwt(x_noisy_test, wavelet(WT.db4), 4);
	p3_test = WaveletsExt.wiggle(z_test)
	Plots.plot!(p3_test, title = "Stationary Transform of Noisy Signal")
	
	Plots.plot(p1_test, p2_test, p3_test, layout = (3,1))
end

# ╔═╡ 356d75f4-6cc1-4062-8ef6-3cc6a9c2d9a7
md"Plot a histogram of wavelet coefficients"

# ╔═╡ 341c2551-b625-47a0-9163-3c0c0e7d4e13
Plots.histogram(vec(abs.(y_test)), legend = false)

# ╔═╡ 9eae2f5c-1f47-43d8-a7ec-0767e50e6a9b
@bind th_method_test Radio(["Hard", "Soft"], default = "Hard")

# ╔═╡ 261c2822-edaf-4c66-a032-c17fc2447627
md"Threshold the coefficients using an arbitrary threshold value"

# ╔═╡ dbed8579-afa4-4a4d-b0bb-bd34877fa272
md"**Select** a threshold value"

# ╔═╡ da070fd3-c11e-4421-80b5-45d764a70deb
@bind th_test Slider(0:0.01:maximum(y_test), default = 0, show_value = true)

# ╔═╡ e49e76e6-7018-4f49-a189-d2fae7df956d
begin
	ŷ_test = deepcopy(y_test)
	ẑ_test = deepcopy(z_test)
	if th_method_test == "Hard"
		threshold!(ŷ_test, HardTH(), th_test);
		threshold!(ẑ_test, SoftTH(), th_test);
	else
		threshold!(ŷ_test, SoftTH(), th_test);
		threshold!(ẑ_test, SoftTH(), th_test);
	end
end;

# ╔═╡ a663045c-fa0f-49fe-88c2-794450cb7806
md"Reconstruct the signal using the thresholded coefficients"

# ╔═╡ 9b4ef541-9a36-4bc0-8654-10ab0a4e63b3
begin
	r1_test = iacwt(ŷ_test)
	r2_test = isdwt(ẑ_test, wavelet(WT.db4))
	Plots.plot(x_test, label = "original", lc = "black", lw=1.5,
		title = "Original vs Denoised Signals")
	Plots.plot!(r1_test, label = "ACWT denoised")
	Plots.plot!(r2_test, label = "SDWT denoised")
	Plots.plot!(x_noisy_test, label = "noisy", lc = "gray", la = 0.3)
end

# ╔═╡ 4669be94-6c4c-42e2-b9d9-2dc98f1bdaea
md"**Calculate the Mean Squared Error between the original signal and the denoised signal**"

# ╔═╡ aebc6084-2807-4818-98a0-119275dc4348
md"MSE between original signal and ACWT denoised signal: $(round(norm(x_test - r1_test)/length(x_test), digits = 4))"

# ╔═╡ 40d91201-6bc8-4baa-8fc9-68efaddcff6e
md"MSE between original signal and SDWT denoised signal: $(round(norm(x_test - r2_test)/length(x_test), digits = 4))"

# ╔═╡ 11e63c9a-6124-4122-9a86-ceed926d25d2
md"# II. Data Setup"

# ╔═╡ d881753b-0432-451b-8de0-38a0b4b4382a
md"**Autorun**: $(@bind autorun CheckBox())

**Note:** Disable before updating parameters! "

# ╔═╡ 8055194b-2e46-4d18-81c0-0c52bc3eb233
md"**Select** a test function"

# ╔═╡ 18b5bbe4-ecdd-4209-a764-7c8b1ecbda61
@bind signal_name Radio(
	[
		"blocks" => "Blocks", 
		"bumps" => "Bumps", 
		"heavy_sine" => "Heavy sine", 
		"doppler" => "Doppler", 
		"quadchirp" => "Quadchirp", 
		"mishmash" => "Mishmash"
	],
	default = "blocks"
)

# ╔═╡ 56ee2c61-d83c-4d76-890a-a9bd0d65cee5
md"**Adjust** the slider to add Gaussian noise to the test signal"

# ╔═╡ c50ac92e-3684-4d0a-a80d-4ee9d74ec992
@bind noise_size Slider(0:0.01:1, default=0.3, show_value=true)

# ╔═╡ e0a96592-5e77-4c29-9744-31369eea8147
md"""
**Select** which type of wavelet basis to use: $(@bind wavelet_type Select(
	["WT.haar", 
	"WT.db1", "WT.db2", "WT.db3", "WT.db4", "WT.db5", 
	"WT.db6", "WT.db7", "WT.db8", "WT.db9", "WT.db10",
	"WT.coif2", "WT.coif4", "WT.coif6", "WT.coif8",
	"WT.sym4", "WT.sym5", "WT.sym6", "WT.sym7", "WT.sym8", "WT.sym9", "WT.sym10",
	"WT.batt2", "WT.batt4", "WT.batt6"],
	default = "WT.haar"
))
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
**Key** in the sample size: $(@bind ss TextField((5,1), default="10"))
"""

# ╔═╡ cd9e259e-8bb3-497b-ac7f-f89a003c8032
begin
	x = testdata[!,signal_name]
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
		X₀ = generatesignals(x, samplesize, 2)
		X = hcat([addnoise(X₀[:,i], noise_size) for i in axes(X₀,2)]...)
	end
end;

# ╔═╡ 6664a859-4980-4b40-8684-83cf2e7db109
md"**Baseline**: No denoising"

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
md"# III. Non-Redundant Transforms"

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
			L=8, 
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
			L=8, 
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
			L=8, 
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
			L=8, 
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
			L=8, 
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
			L=8, 
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
md"### 2. Wavelet Packet Transform - Level 4"

# ╔═╡ cf55c5cb-ead6-40b6-896a-8f7e01613a46
begin
	if autorun
		# Visushrink
		Y["WPT4"] = cat([wpt(X[:,i], wt,4) for i in axes(X,2)]..., dims=2)
		T["WPT4"] = maketree(256, 4, :full)
		σ₁["WPT4"] = [noisest(Y["WPT4"][:,i], false, T["WPT4"]) for i in axes(X,2)]
		σ₂["WPT4"] = [relerrorthreshold(Y["WPT4"][:,i], false, T["WPT4"]) for i in axes(X,2)]
		## bestTH = individual
		X̂["WPT4_vs_ind"], time["WPT4_vs_ind"] = @timed denoiseall(
			Y["WPT4"], 
			:wpt, 
			wt, 
			tree=T["WPT4"], 
			estnoise=σ₁["WPT4"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["WPT4_vs_ind"] = mean(
			[psnr(X̂["WPT4_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT4_vs_ind"] = mean(
			[ssim(X̂["WPT4_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L4", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["WPT4_vs_ind"], 
				mean_psnr["WPT4_vs_ind"],
				mean_ssim["WPT4_vs_ind"]
			]
		)
		## bestTH = average
		X̂["WPT4_vs_avg"], time["WPT4_vs_avg"] = @timed denoiseall(
			Y["WPT4"], 
			:wpt, 
			wt, 
			tree=T["WPT4"], 
			estnoise=σ₁["WPT4"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["WPT4_vs_avg"] = mean(
			[psnr(X̂["WPT4_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT4_vs_avg"] = mean(
			[ssim(X̂["WPT4_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L4", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["WPT4_vs_avg"], 
				mean_psnr["WPT4_vs_avg"],
				mean_ssim["WPT4_vs_avg"]
			]
		)
		## bestTH = median
		X̂["WPT4_vs_med"], time["WPT4_vs_med"] = @timed denoiseall(
			Y["WPT4"], 
			:wpt, 
			wt, 
			tree=T["WPT4"], 
			estnoise=σ₁["WPT4"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["WPT4_vs_med"] = mean(
			[psnr(X̂["WPT4_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT4_vs_med"] = mean(
			[ssim(X̂["WPT4_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L4", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["WPT4_vs_med"], 
				mean_psnr["WPT4_vs_med"],
				mean_ssim["WPT4_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["WPT4_res_ind"], time["WPT4_res_ind"] = @timed denoiseall(
			Y["WPT4"], 
			:wpt, 
			wt, 
			tree=T["WPT4"], 
			estnoise=σ₂["WPT4"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["WPT4_res_ind"] = mean(
			[psnr(X̂["WPT4_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT4_res_ind"] = mean(
			[ssim(X̂["WPT4_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L4", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["WPT4_res_ind"], 
				mean_psnr["WPT4_res_ind"],
				mean_ssim["WPT4_res_ind"]
			]
		)
		## bestTH = average
		X̂["WPT4_res_avg"], time["WPT4_res_avg"] = @timed denoiseall(
			Y["WPT4"], 
			:wpt, 
			wt, 
			tree=T["WPT4"], 
			estnoise=σ₂["WPT4"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["WPT4_res_avg"] = mean(
			[psnr(X̂["WPT4_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT4_res_avg"] = mean(
			[ssim(X̂["WPT4_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L4", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["WPT4_res_avg"], 
				mean_psnr["WPT4_res_avg"],
				mean_ssim["WPT4_res_avg"]
			]
		)
		## bestTH = median
		X̂["WPT4_res_med"], time["WPT4_res_med"] = @timed denoiseall(
			Y["WPT4"], 
			:wpt, 
			wt, 
			tree=T["WPT4"], 
			estnoise=σ₂["WPT4"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["WPT4_res_med"] = mean(
			[psnr(X̂["WPT4_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT4_res_med"] = mean(
			[ssim(X̂["WPT4_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L4", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["WPT4_res_med"], 
				mean_psnr["WPT4_res_med"],
				mean_ssim["WPT4_res_med"]
			]
		)
	end
end;

# ╔═╡ c95ebbed-3d9a-4be2-943b-08c86923ad89
md"### 3. Wavelet Packet Transform - Level 8"

# ╔═╡ 61d745d8-5c74-479b-9698-cd50bb68b3c7
begin
	if autorun
		# Visushrink
		Y["WPT8"] = cat([wpt(X[:,i], wt,8) for i in axes(X,2)]..., dims=2)
		T["WPT8"] = maketree(256, 8, :full)
		σ₁["WPT8"] = [noisest(Y["WPT8"][:,i], false, T["WPT8"]) for i in axes(X,2)]
		σ₂["WPT8"] = [relerrorthreshold(Y["WPT8"][:,i], false, T["WPT8"]) for i in axes(X,2)]
		## bestTH = individual
		X̂["WPT8_vs_ind"], time["WPT8_vs_ind"] = @timed denoiseall(
			Y["WPT8"], 
			:wpt, 
			wt, 
			tree=T["WPT8"], 
			estnoise=σ₁["WPT8"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["WPT8_vs_ind"] = mean(
			[psnr(X̂["WPT8_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT8_vs_ind"] = mean(
			[ssim(X̂["WPT8_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L8", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["WPT8_vs_ind"], 
				mean_psnr["WPT8_vs_ind"],
				mean_ssim["WPT8_vs_ind"]
			]
		)
		## bestTH = average
		X̂["WPT8_vs_avg"], time["WPT8_vs_avg"] = @timed denoiseall(
			Y["WPT8"], 
			:wpt, 
			wt, 
			tree=T["WPT8"], 
			estnoise=σ₁["WPT8"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["WPT8_vs_avg"] = mean(
			[psnr(X̂["WPT8_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT8_vs_avg"] = mean(
			[ssim(X̂["WPT8_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L8", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["WPT8_vs_avg"], 
				mean_psnr["WPT8_vs_avg"],
				mean_ssim["WPT8_vs_avg"]
			]
		)
		## bestTH = median
		X̂["WPT8_vs_med"], time["WPT8_vs_med"] = @timed denoiseall(
			Y["WPT8"], 
			:wpt, 
			wt, 
			tree=T["WPT8"], 
			estnoise=σ₁["WPT8"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["WPT8_vs_med"] = mean(
			[psnr(X̂["WPT8_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT8_vs_med"] = mean(
			[ssim(X̂["WPT8_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L8", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["WPT8_vs_med"], 
				mean_psnr["WPT8_vs_med"],
				mean_ssim["WPT8_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["WPT8_res_ind"], time["WPT8_res_ind"] = @timed denoiseall(
			Y["WPT8"], 
			:wpt, 
			wt, 
			tree=T["WPT8"], 
			estnoise=σ₂["WPT8"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["WPT8_res_ind"] = mean(
			[psnr(X̂["WPT8_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT8_res_ind"] = mean(
			[ssim(X̂["WPT8_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L8", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["WPT8_res_ind"], 
				mean_psnr["WPT8_res_ind"],
				mean_ssim["WPT8_res_ind"]
			]
		)
		## bestTH = average
		X̂["WPT8_res_avg"], time["WPT8_res_avg"] = @timed denoiseall(
			Y["WPT8"], 
			:wpt, 
			wt, 
			tree=T["WPT8"], 
			estnoise=σ₂["WPT8"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["WPT8_res_avg"] = mean(
			[psnr(X̂["WPT8_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT8_res_avg"] = mean(
			[ssim(X̂["WPT8_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L8", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["WPT8_res_avg"], 
				mean_psnr["WPT8_res_avg"],
				mean_ssim["WPT8_res_avg"]
			]
		)
		## bestTH = median
		X̂["WPT8_res_med"], time["WPT8_res_med"] = @timed denoiseall(
			Y["WPT8"], 
			:wpt, 
			wt, 
			tree=T["WPT8"], 
			estnoise=σ₂["WPT8"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["WPT8_res_med"] = mean(
			[psnr(X̂["WPT8_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["WPT8_res_med"] = mean(
			[ssim(X̂["WPT8_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"WPT-L8", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["WPT8_res_med"], 
				mean_psnr["WPT8_res_med"],
				mean_ssim["WPT8_res_med"]
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
md"# IV. Redundant Transforms"

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
			L=8, 
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
			L=8, 
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
			L=8, 
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
			L=8, 
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
			L=8, 
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
			L=8, 
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
md"### 1.2 Autocorrelation Packet Transform - Level 4"

# ╔═╡ c52bd741-00cb-4cf2-97e3-b8dbba3af9ad
begin
	if autorun
		# Visushrink
		Y["ACWPT"] = cat([acwpt(X[:,i], wt, 8) for i in axes(X,2)]..., dims=3)
		σ₁["ACWPT4"] = [
			noisest(Y["ACWPT"][:,:,i], true, T["WPT4"]) for i in axes(X,2)
		]
		σ₂["ACWPT4"] = [
			relerrorthreshold(Y["ACWPT"][:,:,i], true, T["WPT4"]) for i in axes(X,2)
		]
		## bestTH = individual
		X̂["ACWPT4_vs_ind"], time["ACWPT4_vs_ind"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT4"], 
			estnoise=σ₁["ACWPT4"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["ACWPT4_vs_ind"] = mean(
			[psnr(X̂["ACWPT4_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT4_vs_ind"] = mean(
			[ssim(X̂["ACWPT4_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-L4", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["ACWPT4_vs_ind"], 
				mean_psnr["ACWPT4_vs_ind"],
				mean_ssim["ACWPT4_vs_ind"]
			]
		)
		## bestTH = average
		X̂["ACWPT4_vs_avg"], time["ACWPT4_vs_avg"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT4"], 
			estnoise=σ₁["ACWPT4"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["ACWPT4_vs_avg"] = mean(
			[psnr(X̂["ACWPT4_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT4_vs_avg"] = mean(
			[ssim(X̂["ACWPT4_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-L4", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["ACWPT4_vs_avg"], 
				mean_psnr["ACWPT4_vs_avg"],
				mean_ssim["ACWPT4_vs_avg"]
			]
		)
		## bestTH = median
		X̂["ACWPT4_vs_med"], time["ACWPT4_vs_med"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT4"], 
			estnoise=σ₁["ACWPT4"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["ACWPT4_vs_med"] = mean(
			[psnr(X̂["ACWPT4_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT4_vs_med"] = mean(
			[ssim(X̂["ACWPT4_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-L4", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["ACWPT4_vs_med"], 
				mean_psnr["ACWPT4_vs_med"],
				mean_ssim["ACWPT4_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["ACWPT4_res_ind"], time["ACWPT4_res_ind"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT4"], 
			estnoise=σ₂["ACWPT4"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["ACWPT4_res_ind"] = mean(
			[psnr(X̂["ACWPT4_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT4_res_ind"] = mean(
			[ssim(X̂["ACWPT4_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-L4", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["ACWPT4_res_ind"], 
				mean_psnr["ACWPT4_res_ind"],
				mean_ssim["ACWPT4_res_ind"]
			]
		)
		## bestTH = average
		X̂["ACWPT4_res_avg"], time["ACWPT4_res_avg"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT4"], 
			estnoise=σ₂["ACWPT4"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["ACWPT4_res_avg"] = mean(
			[psnr(X̂["ACWPT4_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT4_res_avg"] = mean(
			[ssim(X̂["ACWPT4_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-L4", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["ACWPT4_res_avg"], 
				mean_psnr["ACWPT4_res_avg"],
				mean_ssim["ACWPT4_res_avg"]
			]
		)
		## bestTH = median
		X̂["ACWPT4_res_med"], time["ACWPT4_res_med"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT4"], 
			estnoise=σ₂["ACWPT4"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["ACWPT4_res_med"] = mean(
			[psnr(X̂["ACWPT4_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT4_res_med"] = mean(
			[ssim(X̂["ACWPT4_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-L4", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["ACWPT4_res_med"], 
				mean_psnr["ACWPT4_res_med"],
				mean_ssim["ACWPT4_res_med"]
			]
		)
	end
end;

# ╔═╡ a4b3694e-2ca5-46a5-b1ce-44c2f7ec2006
md"### 1.3 Autocorrelation Packet Transform - Level 8"

# ╔═╡ 250d22dd-1d15-45d4-8aa4-3de1f37b164c
begin
	if autorun
		# Visushrink
		σ₁["ACWPT8"] = [
			noisest(Y["ACWPT"][:,:,i], true, T["WPT8"]) for i in axes(X,2)
		]
		σ₂["ACWPT8"] = [
			relerrorthreshold(Y["ACWPT"][:,:,i], true, T["WPT8"]) for i in axes(X,2)
		]
		## bestTH = individual
		X̂["ACWPT8_vs_ind"], time["ACWPT8_vs_ind"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT8"], 
			estnoise=σ₁["ACWPT8"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["ACWPT8_vs_ind"] = mean(
			[psnr(X̂["ACWPT8_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT8_vs_ind"] = mean(
			[ssim(X̂["ACWPT8_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-L8", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["ACWPT8_vs_ind"], 
				mean_psnr["ACWPT8_vs_ind"],
				mean_ssim["ACWPT8_vs_ind"]
			]
		)
		## bestTH = average
		X̂["ACWPT8_vs_avg"], time["ACWPT8_vs_avg"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT8"], 
			estnoise=σ₁["ACWPT8"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["ACWPT8_vs_avg"] = mean(
			[psnr(X̂["ACWPT8_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT8_vs_avg"] = mean(
			[ssim(X̂["ACWPT8_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-L8", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["ACWPT8_vs_avg"], 
				mean_psnr["ACWPT8_vs_avg"],
				mean_ssim["ACWPT8_vs_avg"]
			]
		)
		## bestTH = median
		X̂["ACWPT8_vs_med"], time["ACWPT8_vs_med"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT8"], 
			estnoise=σ₁["ACWPT8"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["ACWPT8_vs_med"] = mean(
			[psnr(X̂["ACWPT8_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT8_vs_med"] = mean(
			[ssim(X̂["ACWPT8_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-L8", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["ACWPT8_vs_med"], 
				mean_psnr["ACWPT8_vs_med"],
				mean_ssim["ACWPT8_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["ACWPT8_res_ind"], time["ACWPT8_res_ind"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT8"], 
			estnoise=σ₂["ACWPT8"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["ACWPT8_res_ind"] = mean(
			[psnr(X̂["ACWPT8_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT8_res_ind"] = mean(
			[ssim(X̂["ACWPT8_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-L8", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["ACWPT8_res_ind"], 
				mean_psnr["ACWPT8_res_ind"],
				mean_ssim["ACWPT8_res_ind"]
			]
		)
		## bestTH = average
		X̂["ACWPT8_res_avg"], time["ACWPT8_res_avg"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT8"], 
			estnoise=σ₂["ACWPT8"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["ACWPT8_res_avg"] = mean(
			[psnr(X̂["ACWPT8_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT8_res_avg"] = mean(
			[ssim(X̂["ACWPT8_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-L8", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["ACWPT8_res_avg"], 
				mean_psnr["ACWPT8_res_avg"],
				mean_ssim["ACWPT8_res_avg"]
			]
		)
		## bestTH = median
		X̂["ACWPT8_res_med"], time["ACWPT8_res_med"] = @timed denoiseall(
			Y["ACWPT"], 
			:acwpt, 
			wt, 
			tree=T["WPT8"], 
			estnoise=σ₂["ACWPT8"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["ACWPT8_res_med"] = mean(
			[psnr(X̂["ACWPT8_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["ACWPT8_res_med"] = mean(
			[ssim(X̂["ACWPT8_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"ACWPT-L8", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["ACWPT8_res_med"], 
				mean_psnr["ACWPT8_res_med"],
				mean_ssim["ACWPT8_res_med"]
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
			L=8, 
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
			L=8, 
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
			L=8, 
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
			L=8, 
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
			L=8, 
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
			L=8, 
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
md"### 2.2 Stationary Wavelet Packet Transform - Level 4"

# ╔═╡ 6d09613a-5676-4b45-bf44-7e2c40bb71c9
begin
	if autorun
		# Visushrink
		Y["SWPD"] = cat([swpd(X[:,i], wt, 8) for i in axes(X,2)]..., dims=3)
		σ₁["SWPT4"] = [
			noisest(Y["SWPD"][:,:,i], true, T["WPT4"]) for i in axes(X,2)
		]
		σ₂["SWPT4"] = [
			relerrorthreshold(Y["SWPD"][:,:,i], true, T["WPT4"]) for i in axes(X,2)
		]
		## bestTH = individual
		X̂["SWPT4_vs_ind"], time["SWPT4_vs_ind"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT4"], 
			estnoise=σ₁["SWPT4"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["SWPT4_vs_ind"] = mean(
			[psnr(X̂["SWPT4_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT4_vs_ind"] = mean(
			[ssim(X̂["SWPT4_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-L4", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["SWPT4_vs_ind"], 
				mean_psnr["SWPT4_vs_ind"],
				mean_ssim["SWPT4_vs_ind"]
			]
		)
		## bestTH = average
		X̂["SWPT4_vs_avg"], time["SWPT4_vs_avg"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT4"], 
			estnoise=σ₁["SWPT4"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["SWPT4_vs_avg"] = mean(
			[psnr(X̂["SWPT4_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT4_vs_avg"] = mean(
			[ssim(X̂["SWPT4_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-L4", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["SWPT4_vs_avg"], 
				mean_psnr["SWPT4_vs_avg"],
				mean_ssim["SWPT4_vs_avg"]
			]
		)
		## bestTH = median
		X̂["SWPT4_vs_med"], time["SWPT4_vs_med"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT4"], 
			estnoise=σ₁["SWPT4"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["SWPT4_vs_med"] = mean(
			[psnr(X̂["SWPT4_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT4_vs_med"] = mean(
			[ssim(X̂["SWPT4_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-L4", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["SWPT4_vs_med"], 
				mean_psnr["SWPT4_vs_med"],
				mean_ssim["SWPT4_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["SWPT4_res_ind"], time["SWPT4_res_ind"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT4"], 
			estnoise=σ₂["SWPT4"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["SWPT4_res_ind"] = mean(
			[psnr(X̂["SWPT4_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT4_res_ind"] = mean(
			[ssim(X̂["SWPT4_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-L4", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["SWPT4_res_ind"], 
				mean_psnr["SWPT4_res_ind"],
				mean_ssim["SWPT4_res_ind"]
			]
		)
		## bestTH = average
		X̂["SWPT4_res_avg"], time["SWPT4_res_avg"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT4"], 
			estnoise=σ₂["SWPT4"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["SWPT4_res_avg"] = mean(
			[psnr(X̂["SWPT4_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT4_res_avg"] = mean(
			[ssim(X̂["SWPT4_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-L4", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["SWPT4_res_avg"], 
				mean_psnr["SWPT4_res_avg"],
				mean_ssim["SWPT4_res_avg"]
			]
		)
		## bestTH = median
		X̂["SWPT4_res_med"], time["SWPT4_res_med"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT4"], 
			estnoise=σ₂["SWPT4"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["SWPT4_res_med"] = mean(
			[psnr(X̂["SWPT4_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT4_res_med"] = mean(
			[ssim(X̂["SWPT4_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-L4", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["SWPT4_res_med"], 
				mean_psnr["SWPT4_res_med"],
				mean_ssim["SWPT4_res_med"]
			]
		)
	end
end;

# ╔═╡ 25de9867-d737-496c-8e0c-29bcdca898e8
md"### 2.3 Stationary Wavelet Packet Transform - Level 8"

# ╔═╡ 3525a716-0210-4cd2-b780-3f1c27297e09
begin
	if autorun
		# Visushrink
		σ₁["SWPT8"] = [
			noisest(Y["SWPD"][:,:,i], true, T["WPT8"]) for i in axes(X,2)
		]
		σ₂["SWPT8"] = [
			relerrorthreshold(Y["SWPD"][:,:,i], true, T["WPT8"]) for i in axes(X,2)
		]
		## bestTH = individual
		X̂["SWPT8_vs_ind"], time["SWPT8_vs_ind"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT8"], 
			estnoise=σ₁["SWPT8"],
			dnt=vs_dnt, 
			bestTH=nothing
		)
		mean_psnr["SWPT8_vs_ind"] = mean(
			[psnr(X̂["SWPT8_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT8_vs_ind"] = mean(
			[ssim(X̂["SWPT8_vs_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-L8", 
				threshold_method, 
				"VisuShrink",
				"Individual",
				time["SWPT8_vs_ind"], 
				mean_psnr["SWPT8_vs_ind"],
				mean_ssim["SWPT8_vs_ind"]
			]
		)
		## bestTH = average
		X̂["SWPT8_vs_avg"], time["SWPT8_vs_avg"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT8"], 
			estnoise=σ₁["SWPT8"],
			dnt=vs_dnt, 
			bestTH=mean
		)
		mean_psnr["SWPT8_vs_avg"] = mean(
			[psnr(X̂["SWPT8_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT8_vs_avg"] = mean(
			[ssim(X̂["SWPT8_vs_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-L8", 
				threshold_method, 
				"VisuShrink",
				"Average",
				time["SWPT8_vs_avg"], 
				mean_psnr["SWPT8_vs_avg"],
				mean_ssim["SWPT8_vs_avg"]
			]
		)
		## bestTH = median
		X̂["SWPT8_vs_med"], time["SWPT8_vs_med"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT8"], 
			estnoise=σ₁["SWPT8"],
			dnt=vs_dnt, 
			bestTH=median
		)
		mean_psnr["SWPT8_vs_med"] = mean(
			[psnr(X̂["SWPT8_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT8_vs_med"] = mean(
			[ssim(X̂["SWPT8_vs_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-L8", 
				threshold_method, 
				"VisuShrink",
				"Median",
				time["SWPT8_vs_med"], 
				mean_psnr["SWPT8_vs_med"],
				mean_ssim["SWPT8_vs_med"]
			]
		)
		
		# RelErrorShrink
		## bestTH = individual
		X̂["SWPT8_res_ind"], time["SWPT8_res_ind"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT8"], 
			estnoise=σ₂["SWPT8"],
			dnt=res_dnt, 
			bestTH=nothing
		)
		mean_psnr["SWPT8_res_ind"] = mean(
			[psnr(X̂["SWPT8_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT8_res_ind"] = mean(
			[ssim(X̂["SWPT8_res_ind"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-L8", 
				threshold_method, 
				"RelErrorShrink",
				"Individual",
				time["SWPT8_res_ind"], 
				mean_psnr["SWPT8_res_ind"],
				mean_ssim["SWPT8_res_ind"]
			]
		)
		## bestTH = average
		X̂["SWPT8_res_avg"], time["SWPT8_res_avg"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT8"], 
			estnoise=σ₂["SWPT8"],
			dnt=res_dnt, 
			bestTH=mean
		)
		mean_psnr["SWPT8_res_avg"] = mean(
			[psnr(X̂["SWPT8_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT8_res_avg"] = mean(
			[ssim(X̂["SWPT8_res_avg"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-L8", 
				threshold_method, 
				"RelErrorShrink",
				"Average",
				time["SWPT8_res_avg"], 
				mean_psnr["SWPT8_res_avg"],
				mean_ssim["SWPT8_res_avg"]
			]
		)
		## bestTH = median
		X̂["SWPT8_res_med"], time["SWPT8_res_med"] = @timed denoiseall(
			Y["SWPD"], 
			:swpd, 
			wt, 
			tree=T["WPT8"], 
			estnoise=σ₂["SWPT8"],
			dnt=res_dnt, 
			bestTH=median
		)
		mean_psnr["SWPT8_res_med"] = mean(
			[psnr(X̂["SWPT8_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		mean_ssim["SWPT8_res_med"] = mean(
			[ssim(X̂["SWPT8_res_med"][:,i], X₀[:,i]) for i in axes(X,2)]
		)
		push!(
			results, 
			[
				"SWPT-L8", 
				threshold_method, 
				"RelErrorShrink",
				"Median",
				time["SWPT8_res_med"], 
				mean_psnr["SWPT8_res_med"],
				mean_ssim["SWPT8_res_med"]
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

# ╔═╡ 7103af68-9ad7-4b81-8c21-c8b6d7c9f5be
md"## V. Results Analysis"

# ╔═╡ 1682b2eb-b68c-4182-986b-d921fcfc039d
md"""
**Result Summary**  

To refresh and reorder the data frame below, you'll need to uncheck the following checkbox, make the necessary changes, and check the checkbox again.

Show Results $(@bind show_results CheckBox())

Number of results shown:
$(@bind topN Slider(1:17, default=5, show_value=true))
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

		**Note**: You might need to re-run the block below after updating a parameter to display the results>
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
# ╟─ce4bf94e-edef-40c2-8ac5-b741e47a1759
# ╠═f9e001ae-0323-4283-9af1-1a8252b503e7
# ╟─85c26ead-f043-42c8-8245-58c8d03a963d
# ╠═203fb8c8-4358-4908-b616-a691ce329c02
# ╟─46e30602-a850-4175-b4bf-b4ef4b5359aa
# ╟─bc641876-9723-4c68-8221-ad03fb695c82
# ╠═d0c98a14-ad3e-4a0b-b889-a3ea86f888f3
# ╟─b0a28618-7cda-4c05-83d2-b54bbca3f9b5
# ╟─7364da28-6a01-4359-9664-a3097e8bf1f1
# ╟─458a5a1e-c453-4199-befe-2bf4db6825ae
# ╟─97f40df0-9ccc-4e41-bebf-4e7188f33fff
# ╟─7c0dba17-baf3-4b9c-b1c5-486f7e4515f4
# ╠═a184ae65-7947-4ffb-b751-8b6e97a8608b
# ╟─851b04bb-e382-4a0a-98f6-7d4b983ca5ab
# ╟─356d75f4-6cc1-4062-8ef6-3cc6a9c2d9a7
# ╟─341c2551-b625-47a0-9163-3c0c0e7d4e13
# ╟─9eae2f5c-1f47-43d8-a7ec-0767e50e6a9b
# ╟─261c2822-edaf-4c66-a032-c17fc2447627
# ╟─dbed8579-afa4-4a4d-b0bb-bd34877fa272
# ╟─da070fd3-c11e-4421-80b5-45d764a70deb
# ╟─e49e76e6-7018-4f49-a189-d2fae7df956d
# ╟─a663045c-fa0f-49fe-88c2-794450cb7806
# ╟─9b4ef541-9a36-4bc0-8654-10ab0a4e63b3
# ╟─4669be94-6c4c-42e2-b9d9-2dc98f1bdaea
# ╟─aebc6084-2807-4818-98a0-119275dc4348
# ╟─40d91201-6bc8-4baa-8fc9-68efaddcff6e
# ╟─11e63c9a-6124-4122-9a86-ceed926d25d2
# ╟─d881753b-0432-451b-8de0-38a0b4b4382a
# ╟─8055194b-2e46-4d18-81c0-0c52bc3eb233
# ╟─18b5bbe4-ecdd-4209-a764-7c8b1ecbda61
# ╟─56ee2c61-d83c-4d76-890a-a9bd0d65cee5
# ╟─c50ac92e-3684-4d0a-a80d-4ee9d74ec992
# ╟─e0a96592-5e77-4c29-9744-31369eea8147
# ╟─c178527f-96a4-4ac7-bb0c-38b73b38c45b
# ╟─ef3e7b66-fba0-467a-8a73-c9bf31fadbe3
# ╟─cd9e259e-8bb3-497b-ac7f-f89a003c8032
# ╟─3246e8b5-251f-4398-b21c-397341f2542e
# ╟─82e713f8-c870-43d2-a849-e3b401b00459
# ╟─6664a859-4980-4b40-8684-83cf2e7db109
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
# ╟─7103af68-9ad7-4b81-8c21-c8b6d7c9f5be
# ╟─1682b2eb-b68c-4182-986b-d921fcfc039d
# ╟─2aa29e02-6ff5-4226-9be3-367abbdebf22
# ╟─a392aef5-0df4-477c-a40e-48f206ee1c8d
# ╟─0944c544-006a-45cc-b4c9-ad5bf6877ca3
# ╟─9818ca36-0acc-4cc3-924e-8b50813c1da1
# ╟─56c7a1b9-c75f-48d8-a602-c219a2f432af
# ╟─174edf71-3922-4f33-9024-caaaff884889
