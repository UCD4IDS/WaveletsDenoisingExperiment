using 
    Wavelets, 
    WaveletsExt,
    LinearAlgebra,
    Random, 
    CSV, 
    DataFrames, 
    Statistics

function addnoise(x::AbstractArray{<:Number,1}, s::Real=0.1)
    ϵ = randn(length(x))
    ϵ = ϵ/norm(ϵ)
    y = x + ϵ * s * norm(x)
    return y
end

# compute metrics
function computemetrics(x::Array{T}, x₀::Array{T}) where T<:Number
    # PSNR
    r_psnr = [psnr(x[:,i], x₀[:,i]) for i in axes(x₀,2)]
    # SSIM
    r_ssim = [ssim(x[:,i], x₀[:,i]) for i in axes(x₀,2)]
    return mean([r_psnr r_ssim], dims = 1)
end

# single comparison
function singlecomparison(x::Vector{T}, wt::DiscreteWavelet, 
        noise::AbstractFloat, shifts::Integer, samples::Integer) where T<:Number

    n = size(x,1)
    result = Array{AbstractFloat, 2}(undef, (109,3))
    # generate signals
    X₀ = generatesignals(x, samples, shifts)
    X = hcat([addnoise(X₀[:,i], noise) for i in 1:samples]...)

    # NOISY
    result[1,:] = [0 computemetrics(X, X₀)]

    # DWT 
    L = maxtransformlevels(n)
    y = hcat([dwt(X[:,i], wt, L) for i in 1:samples]...)
    σ₁ = [noisest(y[:,i], false) for i in 1:samples]
    σ₂ = [relerrorthreshold(y[:,i], false) for i in 1:samples]
    ## visushrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :dwt, wt, L=L, dnt=VisuShrink(n), estnoise=σ₁, bestTH=nothing
    )
    result[2,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :dwt, wt, L=L, dnt=VisuShrink(n), estnoise=σ₁, bestTH=mean
    )
    result[3,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :dwt, wt, L=L, dnt=VisuShrink(n), estnoise=σ₁, bestTH=median
    )
    result[4,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :dwt, wt, L=L, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=nothing
    )
    result[5,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :dwt, wt, L=L, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=mean
    )
    result[6,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :dwt, wt, L=L, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=median
    )
    result[7,:] = [time computemetrics(X̂, X₀)]
    
    # WPT-L4
    tree = maketree(n, 4, :full)
    y = hcat([wpt(X[:,i], wt, 4) for i in 1:samples]...)
    σ₁ = [noisest(y[:,i], false, tree) for i in 1:samples]
    σ₂ = [relerrorthreshold(y[:,i], false, tree) for i in 1:samples]
    ## visushrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=nothing
    )
    result[8,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=mean
    )
    result[9,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=median
    )
    result[10,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=nothing
    )
    result[11,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=mean
    )
    result[12,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=median
    )
    result[13,:] = [time computemetrics(X̂, X₀)]

    # WPT-L8
    tree = maketree(n, 8, :full)
    y = hcat([wpt(X[:,i], wt, 8) for i in 1:samples]...)
    σ₁ = [noisest(y[:,i], false, tree) for i in 1:samples]
    σ₂ = [relerrorthreshold(y[:,i], false, tree) for i in 1:samples]
    ## visushrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=nothing
    )
    result[14,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=mean
    )
    result[15,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=median
    )
    result[16,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=nothing
    )
    result[17,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=mean
    )
    result[18,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=median
    )
    result[19,:] = [time computemetrics(X̂, X₀)]

    # WPT-BT
    tree = hcat([bestbasistree(X[:,i], wt) for i in 1:samples]...)
    y = hcat([wpt(X[:,i], wt, tree[:,i]) for i in 1:samples]...)
    σ₁ = [noisest(y[:,i], false, tree[:,i]) for i in 1:samples]
    σ₂ = [relerrorthreshold(y[:,i], false, tree[:,i]) for i in 1:samples]
    ## visushrink & bestTH = individual
    X̂, time = @timed hcat([
        denoise(
            y[:,i], :wpt, wt, tree=tree[:,i], dnt=VisuShrink(n), estnoise=σ₁[i]
        ) for i in 1:samples
    ]...)
    result[20,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = mean
    X̂, time = @timed hcat([
        denoise(
            y[:,i], :wpt, wt, tree=tree[:,i], dnt=VisuShrink(n), estnoise=mean(σ₁)
        ) for i in 1:samples
    ]...)
    result[21,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = median
    X̂, time = @timed hcat([
        denoise(
            y[:,i], :wpt, wt, tree=tree[:,i], dnt=VisuShrink(n), estnoise=median(σ₁)
        ) for i in 1:samples
    ]...)
    result[22,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = individual
    X̂, time = @timed hcat([
        denoise(
            y[:,i], :wpt, wt, tree=tree[:,i], dnt=RelErrorShrink(), estnoise=σ₂[i]
        ) for i in 1:samples
    ]...)
    result[23,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = mean
    X̂, time = @timed hcat([
        denoise(
            y[:,i], :wpt, wt, tree=tree[:,i], dnt=RelErrorShrink(), estnoise=mean(σ₂)
        ) for i in 1:samples
    ]...)
    result[24,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = median
    X̂, time = @timed hcat([
        denoise(
            y[:,i], :wpt, wt, tree=tree[:,i], dnt=RelErrorShrink(), estnoise=median(σ₂)
        ) for i in 1:samples
    ]...)
    result[25,:] = [time computemetrics(X̂, X₀)]

    # JBB
    xw = cat([wpd(X[:,i], wt) for i in 1:samples]..., dims=3)
    tree = bestbasistree(xw, JBB())
    y = bestbasiscoef(xw, tree)
    σ₁ = [noisest(y[:,i], false, tree) for i in 1:samples]
    σ₂ = [relerrorthreshold(y[:,i], false, tree) for i in 1:samples]
    ## visushrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=nothing
    )
    result[26,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=mean
    )
    result[27,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=median
    )
    result[28,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=nothing
    )
    result[29,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=mean
    )
    result[30,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=median
    )
    result[31,:] = [time computemetrics(X̂, X₀)]

    # LSDB
    tree = bestbasistree(xw, LSDB())
    y = bestbasiscoef(xw, tree)
    σ₁ = [noisest(y[:,i], false, tree) for i in 1:samples]
    σ₂ = [relerrorthreshold(y[:,i], false, tree) for i in 1:samples]
    ## visushrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=nothing
    )
    result[32,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=mean
    )
    result[33,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=median
    )
    result[34,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=nothing
    )
    result[35,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=mean
    )
    result[36,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=median
    )
    result[37,:] = [time computemetrics(X̂, X₀)]

    # ACWT
    L = maxtransformlevels(n)
    y = cat([acwt(X[:,i], wt, L) for i in 1:samples]..., dims=3)
    σ₁ = [noisest(y[:,:,i], true) for i in 1:samples]
    σ₂ = [relerrorthreshold(y[:,:,i], true, nothing) for i in 1:samples]
    ## visushrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :acwt, wt, L=L, dnt=VisuShrink(n), estnoise=σ₁, bestTH=nothing
    )
    result[38,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :acwt, wt, L=L, dnt=VisuShrink(n), estnoise=σ₁, bestTH=mean
    )
    result[39,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :acwt, wt, L=L, dnt=VisuShrink(n), estnoise=σ₁, bestTH=median
    )
    result[40,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :acwt, wt, L=L, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=nothing
    )
    result[41,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :acwt, wt, L=L, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=mean
    )
    result[42,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :acwt, wt, L=L, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=median
    )
    result[43,:] = [time computemetrics(X̂, X₀)]

    # ACWPT-L4
    tree = maketree(n, 4, :full)
    y = cat([acwpt(X[:,i], wt, 8) for i in 1:samples]..., dims=3)
    σ₁ = [noisest(y[:,:,i], true, tree) for i in 1:samples]
    σ₂ = [relerrorthreshold(y[:,:,i], true, tree) for i in 1:samples]
    ## visushrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=nothing
    )
    result[44,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=mean
    )
    result[45,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=median
    )
    result[46,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=nothing
    )
    result[47,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=mean
    )
    result[48,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=median
    )
    result[49,:] = [time computemetrics(X̂, X₀)]

    # ACWPT-L8
    tree = maketree(n, 8, :full)
    σ₁ = [noisest(y[:,:,i], true, tree) for i in 1:samples]
    σ₂ = [relerrorthreshold(y[:,:,i], true, tree) for i in 1:samples]
    ## visushrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=nothing
    )
    result[50,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=mean
    )
    result[51,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=median
    )
    result[52,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=nothing
    )
    result[53,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=mean
    )
    result[54,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=median
    )
    result[55,:] = [time computemetrics(X̂, X₀)]

    # ACWPT-BT
    tree = bestbasistree(y, BB(redundant=true))
    σ₁ = [noisest(y[:,:,i], true, tree[:,i]) for i in 1:samples]
    σ₂ = [relerrorthreshold(y[:,:,i], true, tree[:,i]) for i in 1:samples]
    ## visushrink & bestTH = individual
    X̂, time = @timed hcat([
        denoise(
            y[:,:,i], :acwpt, wt, tree=tree[:,i], dnt=VisuShrink(n), estnoise=σ₁[i]
        ) for i in 1:samples
    ]...)
    result[56,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = mean
    X̂, time = @timed hcat([
        denoise(
            y[:,:,i], :acwpt, wt, tree=tree[:,i], dnt=VisuShrink(n), estnoise=mean(σ₁)
        ) for i in 1:samples
    ]...)
    result[57,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = median
    X̂, time = @timed hcat([
        denoise(
            y[:,:,i], :acwpt, wt, tree=tree[:,i], dnt=VisuShrink(n), estnoise=median(σ₁)
        ) for i in 1:samples
    ]...)
    result[58,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = individual
    X̂, time = @timed hcat([
        denoise(
            y[:,:,i], :acwpt, wt, tree=tree[:,i], dnt=RelErrorShrink(), estnoise=σ₂[i]
        ) for i in 1:samples
    ]...)
    result[59,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = mean
    X̂, time = @timed hcat([
        denoise(
            y[:,:,i], :acwpt, wt, tree=tree[:,i], dnt=RelErrorShrink(), estnoise=mean(σ₂)
        ) for i in 1:samples
    ]...)
    result[60,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = median
    X̂, time = @timed hcat([
        denoise(
            y[:,:,i], :acwpt, wt, tree=tree[:,i], dnt=RelErrorShrink(), estnoise=median(σ₂)
        ) for i in 1:samples
    ]...)
    result[61,:] = [time computemetrics(X̂, X₀)]

    # ACWPT-JBB
    tree = bestbasistree(y, JBB(redundant=true))
    σ₁ = [noisest(y[:,:,i], true, tree) for i in 1:samples]
    σ₂ = [relerrorthreshold(y[:,:,i], true, tree) for i in 1:samples]
    ## visushrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=nothing
    )
    result[62,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=mean
    )
    result[63,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=median
    )
    result[64,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=nothing
    )
    result[65,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=mean
    )
    result[66,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=median
    )
    result[67,:] = [time computemetrics(X̂, X₀)]

    # ACWPT-LSDB
    tree = bestbasistree(y, LSDB(redundant=true))
    σ₁ = [noisest(y[:,:,i], true, tree) for i in 1:samples]
    σ₂ = [relerrorthreshold(y[:,:,i], true, tree) for i in 1:samples]
    ## visushrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=nothing
    )
    result[68,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=mean
    )
    result[69,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=median
    )
    result[70,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=nothing
    )
    result[71,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=mean
    )
    result[72,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :acwpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=median
    )
    result[73,:] = [time computemetrics(X̂, X₀)]

    # SDWT
    L = maxtransformlevels(n)
    y = cat([sdwt(X[:,i], wt, L) for i in 1:samples]..., dims=3)
    σ₁ = [noisest(y[:,:,i], true) for i in 1:samples]
    σ₂ = [relerrorthreshold(y[:,:,i], true, nothing) for i in 1:samples]
    ## visushrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :sdwt, wt, L=L, dnt=VisuShrink(n), estnoise=σ₁, bestTH=nothing
    )
    result[74,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :sdwt, wt, L=L, dnt=VisuShrink(n), estnoise=σ₁, bestTH=mean
    )
    result[75,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :sdwt, wt, L=L, dnt=VisuShrink(n), estnoise=σ₁, bestTH=median
    )
    result[76,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :sdwt, wt, L=L, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=nothing
    )
    result[77,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :sdwt, wt, L=L, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=mean
    )
    result[78,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :sdwt, wt, L=L, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=median
    )
    result[79,:] = [time computemetrics(X̂, X₀)]

    # SWPT-L4
    tree = maketree(n, 4, :full)
    y = cat([swpd(X[:,i], wt, 8) for i in 1:samples]..., dims=3)
    σ₁ = [noisest(y[:,:,i], true, tree) for i in 1:samples]
    σ₂ = [relerrorthreshold(y[:,:,i], true, tree) for i in 1:samples]
    ## visushrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=nothing
    )
    result[80,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=mean
    )
    result[81,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=median
    )
    result[82,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=nothing
    )
    result[83,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=mean
    )
    result[84,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=median
    )
    result[85,:] = [time computemetrics(X̂, X₀)]

    # SWPT-L8
    tree = maketree(n, 8, :full)
    σ₁ = [noisest(y[:,:,i], true, tree) for i in 1:samples]
    σ₂ = [relerrorthreshold(y[:,:,i], true, tree) for i in 1:samples]
    ## visushrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=nothing
    )
    result[86,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=mean
    )
    result[87,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=median
    )
    result[88,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=nothing
    )
    result[89,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=mean
    )
    result[90,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=median
    )
    result[91,:] = [time computemetrics(X̂, X₀)]

    # SWPT-BT
    tree = bestbasistree(y, BB(redundant=true))
    σ₁ = [noisest(y[:,:,i], true, tree[:,i]) for i in 1:samples]
    σ₂ = [relerrorthreshold(y[:,:,i], true, tree[:,i]) for i in 1:samples]
    ## visushrink & bestTH = individual
    X̂, time = @timed hcat([
        denoise(
            y[:,:,i], :swpd, wt, tree=tree[:,i], dnt=VisuShrink(n), estnoise=σ₁[i]
        ) for i in 1:samples
    ]...)
    result[92,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = mean
    X̂, time = @timed hcat([
        denoise(
            y[:,:,i], :swpd, wt, tree=tree[:,i], dnt=VisuShrink(n), estnoise=mean(σ₁)
        ) for i in 1:samples
    ]...)
    result[93,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = median
    X̂, time = @timed hcat([
        denoise(
            y[:,:,i], :swpd, wt, tree=tree[:,i], dnt=VisuShrink(n), estnoise=median(σ₁)
        ) for i in 1:samples
    ]...)
    result[94,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = individual
    X̂, time = @timed hcat([
        denoise(
            y[:,:,i], :swpd, wt, tree=tree[:,i], dnt=RelErrorShrink(), estnoise=σ₂[i]
        ) for i in 1:samples
    ]...)
    result[95,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = mean
    X̂, time = @timed hcat([
        denoise(
            y[:,:,i], :swpd, wt, tree=tree[:,i], dnt=RelErrorShrink(), estnoise=mean(σ₂)
        ) for i in 1:samples
    ]...)
    result[96,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = median
    X̂, time = @timed hcat([
        denoise(
            y[:,:,i], :swpd, wt, tree=tree[:,i], dnt=RelErrorShrink(), estnoise=median(σ₂)
        ) for i in 1:samples
    ]...)
    result[97,:] = [time computemetrics(X̂, X₀)]

    # SJBB
    tree = bestbasistree(y, JBB(redundant=true))
    σ₁ = [noisest(y[:,:,i], true, tree) for i in 1:samples]
    σ₂ = [relerrorthreshold(y[:,:,i], true, tree) for i in 1:samples]
    ## visushrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=nothing
    )
    result[98,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=mean
    )
    result[99,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=median
    )
    result[100,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=nothing
    )
    result[101,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=mean
    )
    result[102,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=median
    )
    result[103,:] = [time computemetrics(X̂, X₀)]

    # SLSDB
    tree = bestbasistree(y, LSDB(redundant=true))
    σ₁ = [noisest(y[:,:,i], true, tree) for i in 1:samples]
    σ₂ = [relerrorthreshold(y[:,:,i], true, tree) for i in 1:samples]
    ## visushrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=nothing
    )
    result[104,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=mean
    )
    result[105,:] = [time computemetrics(X̂, X₀)]
    ## visushrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ₁, bestTH=median
    )
    result[106,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = individual
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=nothing
    )
    result[107,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = mean
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=mean
    )
    result[108,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink & bestTH = median
    X̂, time = @timed denoiseall(
        y, :swpd, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ₂, bestTH=median
    )
    result[109,:] = [time computemetrics(X̂, X₀)]

    return result
end

# repeated comparisons
function repeatedcomparisons(x::Vector{T}, wt::DiscreteWavelet, 
        noise::AbstractFloat, shifts::Integer, samples::Integer, 
        repeats::Integer) where T<:Number

    results = 
        [singlecomparison(x, wt, noise, shifts, samples) for _ in 1:repeats] |>
        y -> cat(y..., dims = 3) |>
        y -> mean(y, dims = 3) |>
        y -> reshape(y, :, 3) |>
        y -> DataFrame(y, [:Time, :PSNR, :SSIM])

    params = DataFrame(
        transform = ["None"; 
            repeat(["DWT", "WPT-L4", "WPT-L8", "WPT-BT", "JBB", "LSDB"], inner=6);
            repeat(["ACWT", "ACWPT-L4", "ACWPT-L8", "ACWPT-BT", "ACJBB", "ACLSDB"], inner=6);
            repeat(["SDWT", "SWPT-L4", "SWPT-L8", "SWPT-BT", "SJBB", "SLSDB"], inner=6)],
        method = ["None"; 
            repeat(["VisuShrink", "RelErrorShrink"], inner=3, outer=18)],
        THType = ["None"; repeat(["Hard"], inner=108)],
        bestTH = ["None"; 
            repeat(["Individual", "Average", "Median"], outer=36)]
    )
    return [params results]
end


wt = wavelet(WT.db4)
# get signals
wv = CSV.read("./data/wavelet_test_256.csv", DataFrame)

# compute and save results
samples = 100
repeats = 50
results = Dict{String, DataFrame}()
results["blocks"] = repeatedcomparisons(wv.blocks, wt, 0.5, 2, samples, repeats)
CSV.write("./results/blocks.csv", results["blocks"])
results["bumps"] = repeatedcomparisons(wv.bumps, wt, 0.5, 2, samples, repeats)
CSV.write("./results/bumps.csv", results["bumps"])
results["heavysine"] = repeatedcomparisons(wv.heavy_sine, wt, 0.5, 2, samples, repeats)
CSV.write("./results/heavysine.csv", results["heavysine"])
results["doppler"] = repeatedcomparisons(wv.doppler, wt, 0.5, 2, samples, repeats)
CSV.write("./results/doppler.csv", results["doppler"])
results["quadchirp"] = repeatedcomparisons(wv.quadchirp, wt, 0.5, 2, samples, repeats)
CSV.write("./results/quadchirp.csv", results["quadchirp"])
results["mishmash"] = repeatedcomparisons(wv.mishmash, wt, 0.5, 2, samples, repeats)
CSV.write("./results/mishmash.csv", results["mishmash"])

# display results
vscodedisplay(results["blocks"])
vscodedisplay(results["bumps"])
vscodedisplay(results["heavysine"])
vscodedisplay(results["doppler"])
vscodedisplay(results["quadchirp"])
vscodedisplay(results["mishmash"])
