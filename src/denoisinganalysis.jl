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
        point_size=10pt,
        background_color="white",
        key_title_font_size=24pt, key_title_color="black",
        key_label_font_size=20pt, key_label_color="black",
        major_label_font_size=24pt, major_label_color="black",
        minor_label_font_size=20pt, minor_label_color="black"
    )
    Gadfly.push_theme(fonts)

    # plots
    set_default_plot_size(30inch, 20inch)
    psnrplot = plot(
        df[df[:,:transform] .!= "None", :], 
        y=:transform, x=:PSNR, color=:bestTH, ygroup=:method, 
        xintercept=df[df[:, :transform] .== "None", :PSNR],
        Geom.subplot_grid(
            Geom.point, Geom.vline(color=["gray"], size=[3mm]), free_y_axis=true
        ), 
        Guide.title("PSNR")
    )
    ssimplot = plot(
        df[df[:,:transform] .!= "None", :], 
        y=:transform, x=:SSIM, color=:bestTH, ygroup=:method, 
        xintercept=df[df[:, :transform] .== "None", :SSIM],
        Geom.subplot_grid(
            Geom.point, Geom.vline(color="gray", size=[3mm]), free_y_axis=true
        ), 
        Guide.title("SSIM"))
    # stack plots
    result = gridstack([psnrplot ssimplot])
    # add title
    result = vstack(
        compose(
            context(0, 0, 1, 0.04),
            (context(0, 0, 1, 1), text(0.5, 1.0, main, hcenter, vbottom), Compose.fontsize(72pt), fill("black")), 
            (context(), rectangle(), fill(colorant"white"))
        ),                                                          # title
        compose(context(0, 0, 1, 0.96), result))                    # plots
    return result
end

makeplot(blocks, "Blocks")
makeplot(bumps, "Bumps")
makeplot(heavysine, "Heavysine")
makeplot(doppler, "Doppler")
makeplot(quadchirp, "Quadchirp")
makeplot(mishmash, "Mishmash")
