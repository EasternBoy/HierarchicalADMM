import Pkg

Pkg.activate(".")
using StatsPlots, NPZ, Plots, Statistics, Plots


function network_data(D, N, h, f)
    r1 = npzread(joinpath("data","disjoint-problem",string("Max-iter-D=",D,"-N=",N,"-h=",h,"-f=",f,".npz")))
    r2 = npzread(joinpath("data","disjoint-problem",string("Max-com-D=",D,"-N=",N,"-h=",h,"-f=",f,".npz")))
    r3 = npzread(joinpath("data","disjoint-problem",string("Tot-com-D=",D,"-N=",N,"-h=",h,"-f=",f,".npz")))

    return r1,r2,r3    
end


max_iter1, max_com1, tt_com1 = network_data(3,10,19,17)
max_iter2, max_com2, tt_com2 = network_data(3,20,21,17)
max_iter3, max_com3, tt_com3 = network_data(5,20,31,21)


color = Dict("hADMM"=> :blue, "fADMM"=> :green)

tickfont = 16
plt1 = plot(size = (400,600), tickfont = tickfont)
for (key, value) in tt_com1
    if key !== "nADMM"
        boxplot!(plt1, [key], value./100, label="", outliers = false)
        # dotplot!(plt1, [key], value./100, label="",markersize=3)
    end
end
annotate!((0, ylims(plt1)[2], text("x100", :left, tickfont)))
savefig(plt1,joinpath("media","figs","disjoint_problem","Tot-com-3-10.pdf"))


plt2 = plot(size = (400,600), tickfont = tickfont, yticks = [10, 12, 14, 16, 18, 20])
for (key, value) in tt_com2
    if key !== "nADMM"
        boxplot!(plt2, [key], value./100, label="", outliers = false)
        # dotplot!(plt2, [key], value, label="",markersize=3)
    end
end
annotate!((0, ylims(plt2)[2], text("×100", :left, tickfont)))
savefig(plt2,joinpath("media","figs","disjoint_problem","Tot-com-3-20.pdf"))

plt3 = plot(size = (400,600), tickfont = tickfont)
for (key, value) in tt_com3
    if key !== "nADMM"
        boxplot!(plt3, [key], value./100, label="", outliers = false)
        # dotplot!(plt3, [key], value, label="",markersize=3)
    end
end
annotate!((0, ylims(plt3)[2], text("×100", :left, tickfont)))
savefig(plt3,joinpath("media","figs","disjoint_problem","Tot-com-5-20.pdf"))


plt4 = plot(size = (400,600),  tickfont = tickfont)
for (key, value) in max_com1
    if key !== "nADMM"
    boxplot!(plt4, [key], value/100, label="", outliers = false)
    # dotplot!(plt4, [key], value, label="", markersize=3)
    end
end
annotate!((0, ylims(plt4)[2], text("×100", :left, tickfont)))
savefig(plt4,joinpath("media","figs","disjoint_problem","Max-com-3-10.pdf"))

plt5 = plot(size = (400,600),  tickfont = tickfont)
for (key, value) in max_com2
    if key !== "nADMM"
    boxplot!(plt5, [key], value/100, label="", outliers = false)
    # dotplot!(plt5, [key], value, label="", markersize=3)
    end
end
annotate!((0, ylims(plt5)[2], text("×100", :left, tickfont)))
savefig(plt5,joinpath("media","figs","disjoint_problem","Max-com-3-20.pdf"))

plt6 = plot(size = (400,600),  tickfont = tickfont)
for (key, value) in max_com3
    if key !== "nADMM"
    boxplot!(plt6, [key], value/100, label="", outliers = false)
    # dotplot!(plt6, [key], value, label="",markersize=3)
    end
end
annotate!((0, ylims(plt6)[2], text("×100", :left, tickfont)))
savefig(plt6,joinpath("media","figs","disjoint_problem","Max-com-5-20.pdf"))



savefig(figPrime, joinpath("media","figs","disjoint_problem",string("DJ-Prime-Conver D=",string(nD)," N=",string(nN),".pdf")))
savefig(figRes, joinpath("media","figs","disjoint_problem",string("DJ-Res-Conver D=",string(nD)," N=",string(nN),".pdf")))
savefig(figJ, joinpath("media","figs","disjoint_problem",string("DJ-Cost-Conver D=",string(nD)," N=",string(nN),".pdf")))



# max_iter, max_com, tt_com = network_data(5,10,27,20)
# for (key, value) in max_iter
#     med = round(median(value))
#     min_value, min_index = findmin(value)
#     max_value, max_index = findmax(value)
#     println("$key Maximum number of iterations in a node (min) avg. (max): ($min_value) $med ($max_value)")
# end
# println()
# for (key, value) in max_com
#     med = round(median(value))
#     min_value, min_index = findmin(value)
#     max_value, max_index = findmax(value)
#     println("$key Maximum number of scalar variables sent by a node (min) avg. (max): ($min_value) $med ($max_value)")
# end
# println()
# for (key, value) in tt_com
#     med = median(value)
#     min_value, min_index = findmin(value)
#     max_value, max_index = findmax(value)
#     println("$key Total number of scalar variables sent in network (min) avg. (max): ($min_value) $med ($max_value)")
# end


# max_com = npzread(joinpath("code","HADMM-convergence","Data","Max-com-D=5-N=10-h=27-f=20.npz"))
# plt4 = plot(size = (400,600),  tickfont = tickfont)
# for (key, value) in max_com
#     if key !== "nADMM"
#     boxplot!(plt4, [key], value/100, label="")
#     # dotplot!(plt4, [key], value, label="", markersize=3)
#     end
# end
# annotate!((0, ylims(plt4)[2], text("×100", :left, tickfont)))
# png(plt4,joinpath("code","HADMM-convergence","Figs","Max-com-5-20.png"))

# plt5 = plot(size = (400,600),  tickfont = tickfont)
# max_com = npzread(joinpath("code","HADMM-convergence","Data","Max-com-D=5-N=10-h=27-f=20.npz"))
# for (key, value) in max_com
#     if key !== "nADMM"
#     boxplot!(plt5, [key], value/100, label="")
#     # dotplot!(plt5, [key], value, label="", markersize=3)
#     end
# end
# annotate!((0, ylims(plt5)[2], text("×100", :left, tickfont)))
# png(plt5,joinpath("code","HADMM-convergence","Figs","Max-com-3-20.png"))

# plt6 = plot(size = (400,600),  tickfont = tickfont)
# max_com = npzread(joinpath("code","HADMM-convergence","Data","Max-com-D=3-N=10-h=19-f=17.npz"))
# for (key, value) in max_com
#     if key !== "nADMM"
#     boxplot!(plt6, [key], value/100, label="")
#     # dotplot!(plt6, [key], value, label="",markersize=3)
#     end
# end
# annotate!((0, ylims(plt6)[2], text("×100", :left, tickfont)))
# png(plt6,joinpath("code","HADMM-convergence","Figs","Max-com-3-10.png"))