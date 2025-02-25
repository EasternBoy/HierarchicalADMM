push!(LOAD_PATH, ".")

import Pkg
using Pkg

include("myADMM.jl")


global step_arr = Int64[]

nTestTopo = 100
for _ in 1:nTestTopo
    global countID
    global step_arr


    nN   = 10
    nD   = 4
    tol  = 0.01

    root = linknode(string(countID+=1))
    topo_gen!(root, nN, nD)
    assign_var!(root)

    max_iter = 200
    for step in 1:max_iter
        termination = hierarchicalADMM!(root)
        # if step%10 ==  0 println("Step $step: $termination") end
        if termination < tol
            step_arr = [step_arr; step]
            println("Terminate at step $step")
            break
        end
    end

    print_tree(root)
    println()
    countID = -1
end

println(sum(step_arr)/length(step_arr))