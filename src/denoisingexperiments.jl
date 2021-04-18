using 
    Wavelets, 
    WaveletsExt,
    LinearAlgebra,
    Random, 
    CSV, 
    DataFrames, 
    Statistics, 
    Plots

# compute metrics
function computemetrics(x::Array{T}, x₀::Array{T}) where T<:Number
    # L1-relative norm
    r_l1 = [relativenorm(x[:,i], x₀[:,i], 1) for i in axes(x₀,2)]
    # L2-relative norm
    r_l2 = [relativenorm(x[:,i], x₀[:,i], 2) for i in axes(x₀,2)]
    # PSNR
    r_psnr = [psnr(x[:,i], x₀[:,i]) for i in axes(x₀,2)]
    # SNR
    r_snr = [snr(x[:,i], x₀[:,i]) for i in axes(x₀,2)]
    return mean([r_l1 r_l2 r_psnr r_snr], dims = 1)
end

# single comparison
function singlecomparison(x::Vector{T}, wt::DiscreteWavelet, 
        noise::AbstractFloat, shifts::Integer, samples::Integer) where T<:Number

    n = size(x,1)
    result = Array{AbstractFloat, 2}(undef, (127,5))
    # generate signals
    X₀ = generatesignals(x, samples, shifts)
    X = generatesignals(x, samples, shifts, true, noise)

    # noisy
    result[1,:] = [0 computemetrics(X, X₀)]
    # dwt 
    ## visushrink
    L = maxtransformlevels(n)
    y = hcat([dwt(X[:,i], wt, L) for i in 1:samples]...)
    σ = [noisest(y[:,i], false) for i in 1:samples]
    ### hardth
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :dwt, wt, L=L, dnt=VisuShrink(n), estnoise=σ)
    result[2,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :dwt, wt, L=L, dnt=VisuShrink(n), estnoise=σ, bestTH=mean)
    result[3,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :dwt, wt, L=L, dnt=VisuShrink(n), estnoise=σ, bestTH=median)
    result[4,:] = [time computemetrics(X̂, X₀)]
    ### softth
    X̂, time = @timed denoiseall(y, :dwt, wt, L=L, dnt=VisuShrink(SoftTH(),n), estnoise=σ)
    result[5,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :dwt, wt, L=L, dnt=VisuShrink(SoftTH(),n), estnoise=σ, bestTH=mean)
    result[6,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :dwt, wt, L=L, dnt=VisuShrink(SoftTH(),n), estnoise=σ, bestTH=median)
    result[7,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink
    σ = [relerrorthreshold(y[:,i], false) for i in 1:samples]
    ### hardth
    X̂, time = @timed denoiseall(y, :dwt, wt, L=L, dnt=RelErrorShrink(), estnoise=σ)
    result[8,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :dwt, wt, L=L, dnt=RelErrorShrink(), estnoise=σ, bestTH=mean)
    result[9,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :dwt, wt, L=L, dnt=RelErrorShrink(), estnoise=σ, bestTH=median)
    result[10,:] = [time computemetrics(X̂, X₀)]
    ### softth
    σ = [relerrorthreshold(y[:,i], false, nothing, 3) for i in 1:samples]
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :dwt, wt, L=L, dnt=RelErrorShrink(SoftTH()), estnoise=σ)
    result[11,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :dwt, wt, L=L, dnt=RelErrorShrink(SoftTH()), estnoise=σ, bestTH=mean)
    result[12,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :dwt, wt, L=L, dnt=RelErrorShrink(SoftTH()), estnoise=σ, bestTH=median)
    result[13,:] = [time computemetrics(X̂, X₀)]
    
    # wptBT
    tree = hcat([bestbasistree(X[:,i], wt) for i in 1:samples]...)
    y = hcat([wpt(X[:,i], wt, tree[:,i]) for i in 1:samples]...)
    σ = [noisest(y[:,i], false, tree[:,i]) for i in 1:samples]
    ## visushrink
    ### hard th
    #### bestTH = individual
    X̂, time = @timed hcat([denoise(y[:,i], :wpt, wt, tree=tree[:,i], dnt=VisuShrink(n), estnoise=σ[i]) for i in 1:samples]...)
    result[14,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed hcat([denoise(y[:,i], :wpt, wt, tree=tree[:,i], dnt=VisuShrink(n), estnoise=mean(σ)) for i in 1:samples]...)
    result[15,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed hcat([denoise(y[:,i], :wpt, wt, tree=tree[:,i], dnt=VisuShrink(n), estnoise=median(σ)) for i in 1:samples]...)
    result[16,:] = [time computemetrics(X̂, X₀)]
    ### softth
    #### bestTH = individual
    X̂, time = @timed hcat([denoise(y[:,i], :wpt, wt, tree=tree[:,i], dnt=VisuShrink(SoftTH(),n), estnoise=σ[i]) for i in 1:samples]...)
    result[17,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed hcat([denoise(y[:,i], :wpt, wt, tree=tree[:,i], dnt=VisuShrink(SoftTH(),n), estnoise=mean(σ)) for i in 1:samples]...)
    result[18,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed hcat([denoise(y[:,i], :wpt, wt, tree=tree[:,i], dnt=VisuShrink(SoftTH(),n), estnoise=median(σ)) for i in 1:samples]...)
    result[19,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink
    ### hardth
    σ = [relerrorthreshold(y[:,i], false, tree[:,i]) for i in 1:samples]
    #### bestTH = individual
    X̂, time = @timed hcat([denoise(y[:,i], :wpt, wt, tree=tree[:,i], dnt=RelErrorShrink(), estnoise=σ[i]) for i in 1:samples]...)
    result[20,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed hcat([denoise(y[:,i], :wpt, wt, tree=tree[:,i], dnt=RelErrorShrink(), estnoise=mean(σ)) for i in 1:samples]...)
    result[21,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed hcat([denoise(y[:,i], :wpt, wt, tree=tree[:,i], dnt=RelErrorShrink(), estnoise=median(σ)) for i in 1:samples]...)
    result[22,:] = [time computemetrics(X̂, X₀)]
    ### softth
    σ = [relerrorthreshold(y[:,i], false, tree[:,i], 3) for i in 1:samples]
    #### bestTH = individual
    X̂, time = @timed hcat([denoise(y[:,i], :wpt, wt, tree=tree[:,i], dnt=RelErrorShrink(SoftTH()), estnoise=σ[i]) for i in 1:samples]...)
    result[23,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed hcat([denoise(y[:,i], :wpt, wt, tree=tree[:,i], dnt=RelErrorShrink(SoftTH()), estnoise=mean(σ)) for i in 1:samples]...)
    result[24,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed hcat([denoise(y[:,i], :wpt, wt, tree=tree[:,i], dnt=RelErrorShrink(SoftTH()), estnoise=median(σ)) for i in 1:samples]...)
    result[25,:] = [time computemetrics(X̂, X₀)]

    # wptL4
    tree = maketree(n, 4, :full)
    y = hcat([wpt(X[:,i], wt, tree) for i in 1:samples]...)
    σ = [noisest(y[:,i], false, tree) for i in 1:samples]
    ## visushrink
    ### hardth
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ)
    result[26,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ, bestTH=mean)
    result[27,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ, bestTH=median)
    result[28,:] = [time computemetrics(X̂, X₀)]
    ### softth
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(SoftTH(),n), estnoise=σ)
    result[29,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(SoftTH(),n), estnoise=σ, bestTH=mean)
    result[30,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(SoftTH(),n), estnoise=σ, bestTH=median)
    result[31,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink
    ### hardth
    σ = [relerrorthreshold(y[:,i], false, tree) for i in 1:samples]
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ)
    result[32,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ, bestTH=mean)
    result[33,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ, bestTH=median)
    result[34,:] = [time computemetrics(X̂, X₀)]
    ### softth
    σ = [relerrorthreshold(y[:,i], false, tree, 3) for i in 1:samples]
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(SoftTH()), estnoise=σ)
    result[35,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(SoftTH()), estnoise=σ, bestTH=mean)
    result[36,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(SoftTH()), estnoise=σ, bestTH=median)
    result[37,:] = [time computemetrics(X̂, X₀)]

    # wptL8
    tree = maketree(n, 8, :full)
    y = hcat([wpt(X[:,i], wt, tree) for i in 1:samples]...)
    σ = [noisest(y[:,i], false, tree) for i in 1:samples]
    ## visushrink
    ### hardth
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ)
    result[38,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ, bestTH=mean)
    result[39,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ, bestTH=median)
    result[40,:] = [time computemetrics(X̂, X₀)]
    ### softth
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(SoftTH(),n), estnoise=σ)
    result[41,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(SoftTH(),n), estnoise=σ, bestTH=mean)
    result[42,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(SoftTH(),n), estnoise=σ, bestTH=median)
    result[43,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink
    ### hardth
    σ = [relerrorthreshold(y[:,i], false, tree) for i in 1:samples]
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ)
    result[44,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ, bestTH=mean)
    result[45,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ, bestTH=median)
    result[46,:] = [time computemetrics(X̂, X₀)]
    ### softth
    σ = [relerrorthreshold(y[:,i], false, tree, 3) for i in 1:samples]
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(SoftTH()), estnoise=σ)
    result[47,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(SoftTH()), estnoise=σ, bestTH=mean)
    result[48,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(SoftTH()), estnoise=σ, bestTH=median)
    result[49,:] = [time computemetrics(X̂, X₀)]

    # jbb
    xw = wpd(X, wt)
    tree = bestbasistree(xw, JBB())
    y = bestbasiscoef(xw, tree)
    σ = [noisest(y[:,i], false, tree) for i in 1:samples]
    ## visushrink
    ### hardth
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ)
    result[50,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ, bestTH=mean)
    result[51,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ, bestTH=median)
    result[52,:] = [time computemetrics(X̂, X₀)]
    ### softth
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(SoftTH(),n), estnoise=σ)
    result[53,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(SoftTH(),n), estnoise=σ, bestTH=mean)
    result[54,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(SoftTH(),n), estnoise=σ, bestTH=median)
    result[55,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink
    ### hardth
    σ = [relerrorthreshold(y[:,i], false, tree) for i in 1:samples]
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ)
    result[56,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ, bestTH=mean)
    result[57,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ, bestTH=median)
    result[58,:] = [time computemetrics(X̂, X₀)]
    ### softth
    σ = [relerrorthreshold(y[:,i], false, tree, 3) for i in 1:samples]
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(SoftTH()), estnoise=σ)
    result[59,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(SoftTH()), estnoise=σ, bestTH=mean)
    result[60,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(SoftTH()), estnoise=σ, bestTH=median)
    result[61,:] = [time computemetrics(X̂, X₀)]

    # lsdb
    tree = bestbasistree(xw, LSDB())
    y = bestbasiscoef(xw, tree)
    σ = [noisest(y[:,i], false, tree) for i in 1:samples]
    ## visushrink
    ### hardth
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ)
    result[62,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ, bestTH=mean)
    result[63,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ, bestTH=median)
    result[64,:] = [time computemetrics(X̂, X₀)]
    ### softth
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(SoftTH(),n), estnoise=σ)
    result[65,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(SoftTH(),n), estnoise=σ, bestTH=mean)
    result[66,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=VisuShrink(SoftTH(),n), estnoise=σ, bestTH=median)
    result[67,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink
    ### hardth
    σ = [relerrorthreshold(y[:,i], false, tree) for i in 1:samples]
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ)
    result[68,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ, bestTH=mean)
    result[69,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ, bestTH=median)
    result[70,:] = [time computemetrics(X̂, X₀)]
    ### softth
    σ = [relerrorthreshold(y[:,i], false, tree, 3) for i in 1:samples]
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(SoftTH()), estnoise=σ)
    result[71,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(SoftTH()), estnoise=σ, bestTH=mean)
    result[72,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :wpt, wt, tree=tree, dnt=RelErrorShrink(SoftTH()), estnoise=σ, bestTH=median)
    result[73,:] = [time computemetrics(X̂, X₀)]

    # sdwt
    ## visushrink
    ### hardth
    #### bestTH = individual
    L = maxtransformlevels(n)
    y = cat([sdwt(X[:,i], wt, L) for i in 1:samples]..., dims=3)
    σ = [noisest(y[:,:,i], true) for i in 1:samples]
    X̂, time = @timed denoiseall(y, :sdwt, wt, L=L, dnt=VisuShrink(n), estnoise=σ)
    result[74,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :sdwt, wt, L=L, dnt=VisuShrink(n), estnoise=σ, bestTH=mean)
    result[75,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :sdwt, wt, L=L, dnt=VisuShrink(n), estnoise=σ, bestTH=median)
    result[76,:] = [time computemetrics(X̂, X₀)]
    ### softth
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :sdwt, wt, L=L, dnt=VisuShrink(SoftTH(),n), estnoise=σ)
    result[77,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :sdwt, wt, L=L, dnt=VisuShrink(SoftTH(),n), estnoise=σ, bestTH=mean)
    result[78,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :sdwt, wt, L=L, dnt=VisuShrink(SoftTH(),n), estnoise=σ, bestTH=median)
    result[79,:] = [time computemetrics(X̂, X₀)]
    ### steinth
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :sdwt, wt, L=L, dnt=VisuShrink(SteinTH(),n), estnoise=σ)
    result[80,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :sdwt, wt, L=L, dnt=VisuShrink(SteinTH(),n), estnoise=σ, bestTH=mean)
    result[81,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :sdwt, wt, L=L, dnt=VisuShrink(SteinTH(),n), estnoise=σ, bestTH=median)
    result[82,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink
    ### hardth
    σ = [relerrorthreshold(y[:,:,i], true, nothing) for i in 1:samples]
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :sdwt, wt, L=L, dnt=RelErrorShrink(), estnoise=σ)
    result[83,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :sdwt, wt, L=L, dnt=RelErrorShrink(), estnoise=σ, bestTH=mean)
    result[84,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :sdwt, wt, L=L, dnt=RelErrorShrink(), estnoise=σ, bestTH=median)
    result[85,:] = [time computemetrics(X̂, X₀)]
    ### softth
    σ = [relerrorthreshold(y[:,:,i], true, nothing, 3) for i in 1:samples]
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :sdwt, wt, L=L, dnt=RelErrorShrink(SoftTH()), estnoise=σ)
    result[86,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :sdwt, wt, L=L, dnt=RelErrorShrink(SoftTH()), estnoise=σ, bestTH=mean)
    result[87,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :sdwt, wt, L=L, dnt=RelErrorShrink(SoftTH()), estnoise=σ, bestTH=median)
    result[88,:] = [time computemetrics(X̂, X₀)]
    ### steinth
    σ = [relerrorthreshold(y[:,:,i], true, nothing) for i in 1:samples]
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :sdwt, wt, L=L, dnt=RelErrorShrink(SteinTH()), estnoise=σ)
    result[89,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :sdwt, wt, L=L, dnt=RelErrorShrink(SteinTH()), estnoise=σ, bestTH=mean)
    result[90,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :sdwt, wt, L=L, dnt=RelErrorShrink(SteinTH()), estnoise=σ, bestTH=median)
    result[91,:] = [time computemetrics(X̂, X₀)]
    ## sureshrink
    ### hardth
    #### bestTH = individual
    σ = [noisest(y[:,:,i], true) for i in 1:samples]
    X̂, time = @timed hcat([denoise(y[:,:,i], :sdwt, wt, L=L, dnt=SureShrink(y[:,:,i], nothing, HardTH()), estnoise=σ[i]) for i in 1:samples]...)
    result[92,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed hcat([denoise(y[:,:,i], :sdwt, wt, L=L, dnt=SureShrink(y[:,:,i], nothing, HardTH()), estnoise=mean(σ)) for i in 1:samples]...)
    result[93,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed hcat([denoise(y[:,:,i], :sdwt, wt, L=L, dnt=SureShrink(y[:,:,i], nothing, HardTH()), estnoise=median(σ)) for i in 1:samples]...)
    result[94,:] = [time computemetrics(X̂, X₀)]
    ### softth
    #### bestTH = individual
    X̂, time = @timed hcat([denoise(y[:,:,i], :sdwt, wt, L=L, dnt=SureShrink(y[:,:,i], nothing, SoftTH()), estnoise=σ[i]) for i in 1:samples]...)
    result[95,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed hcat([denoise(y[:,:,i], :sdwt, wt, L=L, dnt=SureShrink(y[:,:,i], nothing, SoftTH()), estnoise=mean(σ)) for i in 1:samples]...)
    result[96,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed hcat([denoise(y[:,:,i], :sdwt, wt, L=L, dnt=SureShrink(y[:,:,i], nothing, SoftTH()), estnoise=median(σ)) for i in 1:samples]...)
    result[97,:] = [time computemetrics(X̂, X₀)]
    ### steinth
    #### bestTH = individual
    X̂, time = @timed hcat([denoise(y[:,:,i], :sdwt, wt, L=L, dnt=SureShrink(y[:,:,i], nothing, SteinTH()), estnoise=σ[i]) for i in 1:samples]...)
    result[98,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed hcat([denoise(y[:,:,i], :sdwt, wt, L=L, dnt=SureShrink(y[:,:,i], nothing, SteinTH()), estnoise=mean(σ)) for i in 1:samples]...)
    result[99,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed hcat([denoise(y[:,:,i], :sdwt, wt, L=L, dnt=SureShrink(y[:,:,i], nothing, SteinTH()), estnoise=median(σ)) for i in 1:samples]...)
    result[100,:] = [time computemetrics(X̂, X₀)]

    # sjbb
    y = swpd(X, wt)
    tree = bestbasistree(y, JBB(stationary=true))
    σ = [noisest(y[:,:,i], true, tree) for i in 1:samples]
    ## visushrink
    ### hardth
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :swpd, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ)
    result[101,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :swpd, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ, bestTH=mean)
    result[102,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :swpd, wt, tree=tree, dnt=VisuShrink(n), estnoise=σ, bestTH=median)
    result[103,:] = [time computemetrics(X̂, X₀)]
    ### softth
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :swpd, wt, tree=tree, dnt=VisuShrink(SoftTH(),n), estnoise=σ)
    result[104,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :swpd, wt, tree=tree, dnt=VisuShrink(SoftTH(),n), estnoise=σ, bestTH=mean)
    result[105,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :swpd, wt, tree=tree, dnt=VisuShrink(SoftTH(),n), estnoise=σ, bestTH=median)
    result[106,:] = [time computemetrics(X̂, X₀)]
    ### steinth
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :swpd, wt, tree=tree, dnt=VisuShrink(SteinTH(),n), estnoise=σ)
    result[107,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :swpd, wt, tree=tree, dnt=VisuShrink(SteinTH(),n), estnoise=σ, bestTH=mean)
    result[108,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :swpd, wt, tree=tree, dnt=VisuShrink(SteinTH(),n), estnoise=σ, bestTH=median)
    result[109,:] = [time computemetrics(X̂, X₀)]
    ## relerrorshrink
    ### hardth
    σ = [relerrorthreshold(y[:,:,i], true, tree) for i in 1:samples]
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :swpd, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ)
    result[110,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :swpd, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ, bestTH=mean)
    result[111,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :swpd, wt, tree=tree, dnt=RelErrorShrink(), estnoise=σ, bestTH=median)
    result[112,:] = [time computemetrics(X̂, X₀)]
    ### softth
    σ = [relerrorthreshold(y[:,:,i], true, tree, 3) for i in 1:samples]
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :swpd, wt, tree=tree, dnt=RelErrorShrink(SoftTH()), estnoise=σ)
    result[113,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :swpd, wt, tree=tree, dnt=RelErrorShrink(SoftTH()), estnoise=σ, bestTH=mean)
    result[114,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :swpd, wt, tree=tree, dnt=RelErrorShrink(SoftTH()), estnoise=σ, bestTH=median)
    result[115,:] = [time computemetrics(X̂, X₀)]
    ### steinth
    σ = [relerrorthreshold(y[:,:,i], true, tree) for i in 1:samples]
    #### bestTH = individual
    X̂, time = @timed denoiseall(y, :swpd, wt, tree=tree, dnt=RelErrorShrink(SteinTH()), estnoise=σ)
    result[116,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed denoiseall(y, :swpd, wt, tree=tree, dnt=RelErrorShrink(SteinTH()), estnoise=σ, bestTH=mean)
    result[117,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed denoiseall(y, :swpd, wt, tree=tree, dnt=RelErrorShrink(SteinTH()), estnoise=σ, bestTH=median)
    result[118,:] = [time computemetrics(X̂, X₀)]
    ## sureshrink
    ### hardth
    #### bestTH = individual
    σ = [noisest(y[:,:,i], true) for i in 1:samples]
    X̂, time = @timed hcat([denoise(y[:,:,i], :swpd, wt, tree=tree, dnt=SureShrink(y[:,:,i], tree, HardTH()), estnoise=σ[i]) for i in 1:samples]...)
    result[119,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed hcat([denoise(y[:,:,i], :swpd, wt, tree=tree, dnt=SureShrink(y[:,:,i], tree, HardTH()), estnoise=mean(σ)) for i in 1:samples]...)
    result[120,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed hcat([denoise(y[:,:,i], :swpd, wt, tree=tree, dnt=SureShrink(y[:,:,i], tree, HardTH()), estnoise=median(σ)) for i in 1:samples]...)
    result[121,:] = [time computemetrics(X̂, X₀)]
    ### softth
    #### bestTH = individual
    X̂, time = @timed hcat([denoise(y[:,:,i], :swpd, wt, tree=tree, dnt=SureShrink(y[:,:,i], tree, SoftTH()), estnoise=σ[i]) for i in 1:samples]...)
    result[122,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed hcat([denoise(y[:,:,i], :swpd, wt, tree=tree, dnt=SureShrink(y[:,:,i], tree, SoftTH()), estnoise=mean(σ)) for i in 1:samples]...)
    result[123,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed hcat([denoise(y[:,:,i], :swpd, wt, tree=tree, dnt=SureShrink(y[:,:,i], tree, SoftTH()), estnoise=median(σ)) for i in 1:samples]...)
    result[124,:] = [time computemetrics(X̂, X₀)]
    ### steinth
    #### bestTH = individual
    X̂, time = @timed hcat([denoise(y[:,:,i], :swpd, wt, tree=tree, dnt=SureShrink(y[:,:,i], tree, SteinTH()), estnoise=σ[i]) for i in 1:samples]...)
    result[125,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = average
    X̂, time = @timed hcat([denoise(y[:,:,i], :swpd, wt, tree=tree, dnt=SureShrink(y[:,:,i], tree, SteinTH()), estnoise=mean(σ)) for i in 1:samples]...)
    result[126,:] = [time computemetrics(X̂, X₀)]
    #### bestTH = median
    X̂, time = @timed hcat([denoise(y[:,:,i], :swpd, wt, tree=tree, dnt=SureShrink(y[:,:,i], tree, SteinTH()), estnoise=median(σ)) for i in 1:samples]...)
    result[127,:] = [time computemetrics(X̂, X₀)]

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
        y -> reshape(y, :, 5) |>
        y -> DataFrame(y, [:time, :l1, :l2, :psnr, :snr])

    params = DataFrame(
        transform = ["None"; 
            repeat(["DWT", "BB", "WPT4", "WPT8", "JBB", "LSDB"], inner=12);
            repeat(["SDWT", "SJBB"], inner=27)],
        method = ["None"; 
            repeat(["VisuShrink", "RelErrorShrink"], inner=6, outer=6);
            repeat(["VisuShrink", "RelErrorShrink", "SUREShrink"], inner=9, outer=2)],
        THType = ["None"; 
            repeat(["Hard", "Soft"], inner=3, outer=12);
            repeat(["Hard", "Soft", "Stein"], inner=3, outer=6)],
        bestTH = ["None"; 
            repeat(["Individual", "Average", "Median"], outer=42)]
    )
    return [params results]
end


wt = wavelet(WT.db4)
# get signals
td = CSV.read("thunder256.csv", DataFrame).value
wv = CSV.read("wavelet_test_256.csv", DataFrame)

# compute and save results
samples = 100
repeats = 50
results = Dict{String, DataFrame}()
results["thunder"] = repeatedcomparisons(td, wt, 0.003, 2, samples, repeats)
CSV.write("./tests/results/thunder.csv", results["thunder"])
results["blocks"] = repeatedcomparisons(wv.blocks, wt, 1.25, 2, samples, repeats)
CSV.write("./tests/results/blocks.csv", results["blocks"])
results["bumps"] = repeatedcomparisons(wv.bumps, wt, 0.5, 2, samples, repeats)
CSV.write("./tests/results/bumps.csv", results["bumps"])
results["heavysine"] = repeatedcomparisons(wv.heavy_sine, wt, 0.6, 2, samples, repeats)
CSV.write("./tests/results/heavysine.csv", results["heavysine"])
results["doppler"] = repeatedcomparisons(wv.doppler, wt, 0.08, 2, samples, repeats)
CSV.write("./tests/results/doppler.csv", results["doppler"])
results["quadchirp"] = repeatedcomparisons(wv.quadchirp, wt, 0.3, 2, samples, repeats)
CSV.write("./tests/results/quadchirp.csv", results["quadchirp"])
results["mishmash"] = repeatedcomparisons(wv.mishmash, wt, 0.7, 2, samples, repeats)
CSV.write("./tests/results/mishmash.csv", results["mishmash"])

# display results
vscodedisplay(results["thunder"])
vscodedisplay(results["blocks"])
vscodedisplay(results["bumps"])
vscodedisplay(results["heavysine"])
vscodedisplay(results["doppler"])
vscodedisplay(results["quadchirp"])
vscodedisplay(results["mishmash"])
