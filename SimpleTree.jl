push!(LOAD_PATH, ".")


import Pkg
using Pkg
Pkg.activate(@__DIR__)


include("node.jl")


nlocal = 2

D = 2
rootnode = node(0, 1,-1, [1,2,3], zeros(nlocal), 0.)
node11   = node(1, 1, 0, [1,2],   zeros(nlocal), 1.1)
node12   = node(1, 2, 0, [3,4],   zeros(nlocal), 1.2)
node13   = node(1, 3, 0, [5,6],   zeros(nlocal), 1.3)
node21   = node(2, 1, 1, Int64[], zeros(nlocal), 2.1)
node22   = node(2, 2, 1, Int64[], zeros(nlocal), 2.2)
node23   = node(2, 3, 2, Int64[], zeros(nlocal), 2.3)
node24   = node(2, 4, 2, Int64[], zeros(nlocal), 2.4)
node25   = node(2, 5, 3, Int64[], zeros(nlocal), 2.5)
node26   = node(2, 6, 3, Int64[], zeros(nlocal), 2.6)




tree = [[node11, node12, node13], 
        [node21, node22, node23, node24, node25, node26]];

for d in D:-1:1
    for i in 1:length(tree[d])
        if d == 1
            push!(rootnode.couple_state, vect(tree[d][i].couple_state))
            push!(rootnode.dual_state, vect(tree[d][i].dual_state))
        else
            p = tree[d][i].parent
            push!(tree[d-1][p].couple_state, vect(tree[d][i].couple_state))
            push!(tree[d-1][p].dual_state, vect(tree[d][i].dual_state))
        end
    end
end

tTree = (rootnode, tree)



for k = 1:20
    println("Step $k")
    #Forward
    prox!(rootnode, tTree)
    for d in 1:D
        for i in 1:length(tree[d])
            prox!(tree[d][i], tTree)
        end
    end

    #Backward
    for d in D:-1:1
        for i in 1:length(tree[d])    
            if d == 1
                res = vect(tree[d][i].couple_state) - rootnode.couple_state[i]
                rootnode.dual_state[i] = rootnode.dual_state[i] + res
                println(round(norm(res), digits = 3))
            else
                p   = tree[d][i].parent
                ic  = indexin(tree[d][i].index, tree[d-1][p].children)[1]
                res = vect(tree[d][i].couple_state) - tree[d-1][p].couple_state[ic]
                tree[d-1][p].dual_state[ic]  = tree[d-1][p].dual_state[ic] + res
                println(round(norm(res), digits = 3))
            end
        end
    end
end