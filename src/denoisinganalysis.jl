using CSV, DataFrames, Gadfly, Compose

# import files
thunder = CSV.read("./tests/results/thunder.csv", DataFrame)
blocks = CSV.read("./tests/results/blocks.csv", DataFrame)
bumps = CSV.read("./tests/results/bumps.csv", DataFrame)
heavysine = CSV.read("./tests/results/heavysine.csv", DataFrame)
doppler = CSV.read("./tests/results/doppler.csv", DataFrame)
quadchirp = CSV.read("./tests/results/quadchirp.csv", DataFrame)
mishmash = CSV.read("./tests/results/mishmash.csv", DataFrame)

# build plots
# TODO: use Geom.vline for threshold lines to represent the values of noisy signals
function makeplot(df::DataFrame, main::AbstractString)
    # start theme
    fonts = Theme(
        point_size=12pt,
        background_color="black",
        key_title_font_size=20pt, key_title_color="white",
        key_label_font_size=18pt, key_label_color="white",
        major_label_font_size=20pt, major_label_color="white",
        minor_label_font_size=18pt, minor_label_color="white"
    )
    Gadfly.push_theme(fonts)

    # plots
    set_default_plot_size(50inch, 30inch)
    l1plot = plot(df, y=:transform, x=:l1, color=:THType, shape=:bestTH, 
        ygroup=:method, Geom.subplot_grid(Geom.point, free_y_axis=true), 
        Guide.title("L1"))
    l2plot = plot(df, y=:transform, x=:l2, color=:THType, shape=:bestTH, 
        ygroup=:method, Geom.subplot_grid(Geom.point, free_y_axis=true), 
        Guide.title("L2"))
    psnrplot = plot(df, y=:transform, x=:psnr, color=:THType, shape=:bestTH, 
        ygroup=:method, Geom.subplot_grid(Geom.point, free_y_axis=true), 
        Guide.title("PSNR"))
    snrplot = plot(df, y=:transform, x=:snr, color=:THType, shape=:bestTH, 
        ygroup=:method, Geom.subplot_grid(Geom.point, free_y_axis=true), 
        Guide.title("SNR"))
    # stack plots
    result = gridstack([l1plot l2plot psnrplot snrplot])
    # add title
    result = vstack(
        compose(context(0, 0, 1, 0.04), text(0.5, 1.0, main, hcenter, vbottom), 
            Compose.fontsize(72pt), fill(colorant"white")),         # title
        compose(context(0, 0, 1, 0.96), result))                    # plots
    return result
end

makeplot(thunder, "Thunder")
makeplot(blocks, "Blocks")
makeplot(bumps, "Bumps")
makeplot(heavysine, "Heavysine")
makeplot(doppler, "Doppler")
makeplot(quadchirp, "Quadchirp")
makeplot(mishmash, "Mishmash")

"""
Notes:
Threshold determination methods (TODO):
    - RelErrorShrink
    - VisuShrink
    - SUREShrink
    - (MiniMaxShrink)
Threshold methods:
    - Hard
    - Soft
    - Stein
    - (Semisoft)
    - (Improved) - Wang, Dai
Results:
    - SUREShrink performs best for noisy signals (quadchirp, mishmash), but it's 
        only slightly better than RelErrorShrink.
    - Opposite is true for VisuShrink. It does extremely bad with noisy signals.
    - VisuShrink and SUREShrink does significantly better than RelErrorShrink
        for smooth signals (heavysine, doppler)
    - SUREShrink does not perform well when signal is sparse (thunder)
    - SJBB with VisuShrink works best with signals that are smooth all the way
        (bumps, heavysine)
    - Autocorrelation transforms with RelErrorShrink does significantly better
        than all other methods for quadchirp. For mishmash SJBB with SUREShrink 
        managed to get similar results
    - SDWT was somehow never the best performer to any signal. In fact, it 
        produced the worst results when used with RelErrorShrink. The only times
        it works well are when used with VisuShrink/SUREShrink on smooth 
        signals (bumps, heavysine, doppler)
# TODO:
    - Threshold determination is the most challenging step and can have the 
        largest effect on denoising. Donoho & Johnstone's papers experimented
        on signals with noise ~ N(0,1). VisuShrink and RelErrorShrink are only 
        ones that have solved this problem.
    - Wang & Dai's method might improve denoising on noisy signals, but have 
        yet been tested.
    - Zhang et al.'s method also seems promising, but is a bit more complicated.
        This method was also originally built for 2D-signal denoising.
    - He et al's adaptive threshold determination strategy can also be useful,
        but requires quite a bit of tweaks in algorithm since `t` is constantly
        changing.
    - Schlenke et al's method of modifying estimated noise variance by windows
        also seem plausible, but once again the thresholds are constantly 
        changing.
    - Note that all above methods are meant for denoising individual signals,
        not by group.
"""