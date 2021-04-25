using CSV, DataFrames, Gadfly, Compose

# import files
blocks = CSV.read("./results/blocks.csv", DataFrame)
bumps = CSV.read("./results/bumps.csv", DataFrame)
heavysine = CSV.read("./results/heavysine.csv", DataFrame)
doppler = CSV.read("./results/doppler.csv", DataFrame)
quadchirp = CSV.read("./results/quadchirp.csv", DataFrame)
mishmash = CSV.read("./results/mishmash.csv", DataFrame)

# build plots
function makeplot(df::DataFrame, main::AbstractString)
    # start theme
    fonts = Theme(
        point_size=4pt,
        background_color="white",
        key_title_font_size=12pt, key_title_color="black",
        key_label_font_size=10pt, key_label_color="black",
        major_label_font_size=12pt, major_label_color="black",
        minor_label_font_size=10pt, minor_label_color="black"
    )
    Gadfly.push_theme(fonts)

    # plots
    set_default_plot_size(10inch, 7inch)
    psnrplot = plot(
        df[df[:,:transform] .!= "None", :], 
        y=:transform, x=:PSNR, color=:bestTH, ygroup=:method, 
        xintercept=df[df[:, :transform] .== "None", :PSNR],
        Geom.subplot_grid(
            Geom.point, Geom.vline(color=["gray"], size=[2mm]), free_y_axis=true
        ), 
        Guide.title("Peak Signal-to-Noise Ratio (PSNR)")
    )
    ssimplot = plot(
        df[df[:,:transform] .!= "None", :], 
        y=:transform, x=:SSIM, color=:bestTH, ygroup=:method, 
        xintercept=df[df[:, :transform] .== "None", :SSIM],
        Geom.subplot_grid(
            Geom.point, Geom.vline(color="gray", size=[2mm]), free_y_axis=true
        ), 
        Guide.title("Structural Similarity Index Measure (SSIM)"))
    # stack plots
    result = gridstack([psnrplot ssimplot])
    # add title
    result = vstack(
        compose(
            context(0, 0, 1, 0.04),
            (context(0, 0, 1, 1), text(0.5, 1.0, main, hcenter, vbottom), Compose.fontsize(20pt), fill("black")), 
            (context(), rectangle(), fill(colorant"white"))
        ),                                                          # title
        compose(context(0, 0, 1, 0.96), result))                    # plots
    return result
end

blocksplot = makeplot(blocks, "Blocks")
bumpsplot = makeplot(bumps, "Bumps")
heavysineplot = makeplot(heavysine, "Heavysine")
dopplerplot = makeplot(doppler, "Doppler")
quadchirpplot = makeplot(quadchirp, "Quadchirp")
mishmashplot = makeplot(mishmash, "Mishmash")

draw(SVG("./figures/blocks.svg", 10inch, 7inch), blocksplot)
draw(SVG("./figures/bumps.svg", 10inch, 7inch), bumpsplot)
draw(SVG("./figures/heavysine.svg", 10inch, 7inch), heavysineplot)
draw(SVG("./figures/doppler.svg", 10inch, 7inch), dopplerplot)
draw(SVG("./figures/quadchirp.svg", 10inch, 7inch), quadchirpplot)
draw(SVG("./figures/mishmash.svg", 10inch, 7inch), mishmashplot)