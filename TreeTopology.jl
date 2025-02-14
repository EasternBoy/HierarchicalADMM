push!(LOAD_PATH, ".")


import Pkg
using Pkg
Pkg.activate(@__DIR__)


include("node.jl")


nlocal = 2

D = 3
root     = node(0, 1,-1, [1,2,3], zeros(nlocal), 0.)
node11   = node(1, 1, 0, [1,2],   zeros(nlocal), 1.1)
node12   = node(1, 2, 0, Int64[], zeros(nlocal), 1.2)
node13   = node(1, 3, 0, [3,4],   zeros(nlocal), 1.3)
node21   = node(2, 1, 1, Int64[], zeros(nlocal), 2.1)
node22   = node(2, 2, 1, Int64[], zeros(nlocal), 2.2)
node23   = node(2, 3, 3, [1,2,3], zeros(nlocal), 2.3)
node24   = node(2, 4, 3, Int64[], zeros(nlocal), 2.4)
node31   = node(3, 1, 3, Int64[], zeros(nlocal), 3.1)
node32   = node(3, 2, 3, Int64[], zeros(nlocal), 3.2)
node33   = node(3, 3, 3, Int64[], zeros(nlocal), 3.3)


tree = [[node11, node12, node13], 
        [node21, node22, node23, node24], 
        [node31, node32, node33]];

for d in D:-1:1
    for i in 1:length(tree[d])
        if d == 1
            push!(root.couple_state, vect(tree[d][i].couple_state))
            push!(root.dual_state, vect(tree[d][i].dual_state))
        else
            p = tree[d][i].parent
            push!(tree[d-1][p].couple_state, vect(tree[d][i].couple_state))
            push!(tree[d-1][p].dual_state, vect(tree[d][i].dual_state))
        end
    end
end

tTree = (root, tree)



for k = 1:100
    println("Step $k")
    #Forward
    prox!(root, tTree)
    for d in 1:D
        for i in 1:length(tree[d])
            prox!(tree[d][i], tTree)
        end
    end

    #Backward
    for d in D:-1:1
        for i in 1:length(tree[d])    
            if d == 1
                res = root.couple_state[i] - vect(tree[d][i].couple_state)
                root.dual_state[i] = root.dual_state[i] + res
                println(round(norm(res), digits = 3))
            else
                p   = tree[d][i].parent
                ic  = indexin(tree[d][i].index, tree[d-1][p].children)[1]
                res = tree[d-1][p].couple_state[ic] - vect(tree[d][i].couple_state)
                tree[d-1][p].dual_state[ic]  = tree[d-1][p].dual_state[ic] + res
                println(round(norm(res), digits = 3))
            end
        end
    end
end