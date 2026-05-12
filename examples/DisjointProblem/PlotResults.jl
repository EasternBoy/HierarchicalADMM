import Pkg

Pkg.activate(joinpath(@__DIR__, "..", ".."))
using StatsPlots, NPZ, Plots, Statistics, CSV, DataFrames


function network_data(D, N, h, f)
    f1 = joinpath(@__DIR__, "..", "..", "data","disjoint-problem",string("Max-iter-D=",D,"-N=",N,"-h=",h,"-f=",f,".npz"))
    f2 = joinpath(@__DIR__, "..", "..", "data","disjoint-problem",string("Max-com-D=",D,"-N=",N,"-h=",h,"-f=",f,".npz"))
    f3 = joinpath(@__DIR__, "..", "..", "data","disjoint-problem",string("Tot-com-D=",D,"-N=",N,"-h=",h,"-f=",f,".npz"))

    if !isfile(f1) || !isfile(f2) || !isfile(f3)
        println("Warning: Missing .npz files for D=$D, N=$N, h=$h, f=$f")
        return Dict(), Dict(), Dict()
    end

    r1 = npzread(f1)
    r2 = npzread(f2)
    r3 = npzread(f3)

    return r1,r2,r3    
end


# max_iter1, max_com1, tt_com1 = network_data(3,20,21,17)
max_iter2, max_com2, tt_com2 = network_data(3,20,22,17)
# max_iter3, max_com3, tt_com3 = network_data(5,30,123,50)


color = Dict("hADMM"=> :blue, "fADMM"=> :green)

tickfont = 16
width = 2
viol = true
# plt1 = plot(size = (400,600), tickfont = tickfont)
# for (key, value) in tt_com1
#     if key !== "nADMM"
#         x = fill(key, length(value))
#         if viol == true violin!(plt1, x, value./100, label="", color=:red, fillalpha=0.25, linealpha=0, outliers = false) end
#         boxplot!(plt1, x, value./100, label="", outliers = false, linewidth = width)
#         # dotplot!(plt1, [key], value./100, label="",markersize=3)
#     end
# end
# annotate!((0, ylims(plt1)[2], text("x100", :left, tickfont)))
# savefig(plt1,joinpath("media","figs","disjoint_problem","Tot-com-3-10.pdf"))


plt2 = plot(size = (400,600), tickfont = tickfont, yticks = [10, 12, 14, 16, 18, 20])
for (key, value) in tt_com2
    if key !== "nADMM"
        x = fill(key, length(value))
        if viol == true violin!(plt2, x, value./100, label="", color=:red, fillalpha=0.25, linealpha=0, outliers = false) end
        boxplot!(plt2, x, value./100, label="", outliers = false, linewidth = width)
        # dotplot!(plt2, [key], value, label="",markersize=3)
    end
end
annotate!((0, ylims(plt2)[2], text("×100", :left, tickfont)))
savefig(plt2,joinpath(@__DIR__, "..", "..", "media","figs","disjoint_problem","Tot-com-3-20.pdf"))

# plt3 = plot(size = (400,600), tickfont = tickfont)
# for (key, value) in tt_com3
#     if key !== "nADMM"
#         x = fill(key, length(value))
#         if viol == true violin!(plt3, x, value./100, label="", color=:red, fillalpha=0.25, linealpha=0, outliers = false) end
#         boxplot!(plt3, [key], value./100, label="", outliers = false, linewidth = width)
#         # dotplot!(plt3, [key], value, label="",markersize=3)
#     end
# end
# annotate!((0, ylims(plt3)[2], text("×100", :left, tickfont)))
# savefig(plt3,joinpath("media","figs","disjoint_problem","Tot-com-5-20.pdf"))


# plt4 = plot(size = (400,600),  tickfont = tickfont)
# for (key, value) in max_com1
#     if key !== "nADMM"
#         x = fill(key, length(value))
#         if viol == true violin!(plt4, x, value./100, label="", color=:red, fillalpha=0.25, linealpha=0, outliers = false) end
#         boxplot!(plt4, [key], value/100, label="", outliers = false, linewidth = width)
#     end
# end
# annotate!((0, ylims(plt4)[2], text("×100", :left, tickfont)))
# savefig(plt4,joinpath("media","figs","disjoint_problem","Max-com-3-10.pdf"))

plt5 = plot(size = (400,600),  tickfont = tickfont)
for (key, value) in max_com2
    if key !== "nADMM"
        x = fill(key, length(value))
        if viol == true violin!(plt5, x, value./100, label="", color=:red, fillalpha=0.25, linealpha=0, outliers = false) end
        boxplot!(plt5, [key], value/100, label="", outliers = false, linewidth = width)
    end
end
annotate!((0, ylims(plt5)[2], text("×100", :left, tickfont)))
savefig(plt5,joinpath(@__DIR__, "..", "..", "media","figs","disjoint_problem","Max-com-3-20.pdf"))

# plt6 = plot(size = (400,600),  tickfont = tickfont)
# for (key, value) in max_com3
#     if key !== "nADMM"
#         x = fill(key, length(value))
#         if viol == true violin!(plt6, x, value./100, label="", color=:red, fillalpha=0.25, linealpha=0, outliers = false) end
#         boxplot!(plt6, [key], value/100, label="", outliers = false, linewidth = width)
#     end
# end
# annotate!((0, ylims(plt6)[2], text("×100", :left, tickfont)))
# savefig(plt6,joinpath("media","figs","disjoint_problem","Max-com-5-20.pdf"))


# Function to calculate optimality gap from trajectories
function calculate_metrics(df, topo, alg, opt_val)
    topo_df = filter(row -> row.topology == topo, df)
    
    obj_col = Symbol(string(alg, "_objective"))
    # Only filter for this specific algorithm's data
    valid_idx = .!ismissing.(topo_df[!, obj_col])
    
    iters = topo_df.iteration[valid_idx]
    obj_vals = collect(skipmissing(topo_df[!, obj_col]))
    
    # Calculate optimality gap (%)
    opt_gap = abs.(obj_vals .- opt_val) ./ abs(opt_val) .* 100
    
    return iters, opt_gap
end

# Function to get total communication from CSV
function get_total_communication(df, topo, alg)
    topo_df = filter(row -> row.topology == topo, df)
    
    com_col = Symbol(string(alg, "_total_communication"))
    # Only filter for this specific algorithm's data
    valid_idx = .!ismissing.(topo_df[!, com_col])
    
    iters = topo_df.iteration[valid_idx]
    total_com = collect(skipmissing(topo_df[!, com_col]))
    
    return iters, total_com
end

# Function to plot trajectories with dual y-axes from CSV file
function plot_trajectories(D, N)
    file_path = joinpath(@__DIR__, "..", "..", "data", "disjoint-problem", string("trajectories-D=", D, "-N=", N, ".csv"))
    
    if !isfile(file_path)
        println("File not found: ", file_path)
        println("Please run main.jl first to generate the trajectories CSV file.")
        return
    end
    
    println("Reading trajectories from: ", file_path)
    df = CSV.read(file_path, DataFrame)
    
    unique_topos = unique(df.topology)
    colors = [:blue, :green, :red, :purple, :orange, :cyan, :brown, :magenta, :black]
    
    # Plot for each algorithm with dual y-axes
    for alg in ["hADMM", "fADMM", "nADMM"]
        # Create main plot for optimality gap (left y-axis)
        plt = plot(framestyle=:box, tickfont=14, guidefont=14, legendfontsize=8,
                   xlabel="Iteration", ylabel="Optimality Gap (%)",
                   yscale=:log10, legend=:topright, size=(800, 600))
        
        # Create twin axis for total communication (right y-axis)
        plt2 = twinx(plt)
        plot!(plt2, ylabel="Total Communication", tickfont=14, guidefont=14,
              yscale=:log10, legend=false)
        
        for (i, topo) in enumerate(unique_topos)
            c = colors[mod1(i, length(colors))]
            
            # Get optimal value for this topology
            topo_df = filter(row -> row.topology == topo, df)
            if isempty(topo_df)
                continue
            end
            opt_val = topo_df.optimal_value[1]
            
            # Calculate optimality gap
            iters, opt_gap = calculate_metrics(df, topo, alg, opt_val)
            
            # Get total communication from CSV
            _, total_com = get_total_communication(df, topo, alg)
            
            if !isempty(iters)
                # Plot optimality gap on left axis
                plot!(plt, iters, opt_gap, color=c, linewidth=2, alpha=0.7, 
                      label="Topo $topo")
                
                # Plot total communication on right axis
                plot!(plt2, iters, total_com, color=c, linewidth=2, alpha=0.7,
                      label="")
            end
        end
        
        # Save plot
        savefig(plt, joinpath(@__DIR__, "..", "..", "media", "figs", "disjoint_problem",
                string("Trajectory-DualAxis-", alg, "-D=", D, "-N=", N, ".pdf")))
        
        println("Saved dual-axis plot for $alg")
    end
end

# Generate trajectory plots
println("\nGenerating trajectory plots from CSV...")
plot_trajectories(3, 20)
println("\nAll plots generated successfully!")