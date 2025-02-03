push!(LOAD_PATH, ".")


import Pkg
using Pkg
Pkg.activate(@__DIR__)


include("node.jl")


nlocal = 2

D = 3
rootnode = node(0,-1, [1,2,3], zeros(nlocal), 0.)
node11   = node(1, 0, [1,2],   zeros(nlocal), 1.1)
node12   = node(1, 0, Int64[], zeros(nlocal), 1.2)
node13   = node(1, 0, [3,4],   zeros(nlocal), 1.3)
node21   = node(2, 1, Int64[], zeros(nlocal), 2.1)
node22   = node(2, 1, Int64[], zeros(nlocal), 2.2)
node23   = node(2, 3, [1,2,3], zeros(nlocal), 2.3)
node24   = node(2, 3, Int64[], zeros(nlocal), 2.4)
node31   = node(3, 3, Int64[], zeros(nlocal), 3.1)
node32   = node(3, 3, Int64[], zeros(nlocal), 3.2)
node33   = node(3, 3, Int64[], zeros(nlocal), 3.3)


tree = [[node11, node12, node13], 
        [node21, node22, node23, node24], 
        [node31, node32, node33]]

for d in D:-1:1
    for i in 1:length(tree[d])
        p = tree[d][i].parent
        if d == 1
            push!(rootnode.couple_state, vect(tree[d][i].couple_state))
            push!(rootnode.dual_state, vect(tree[d][i].dual_state))
        else
            push!(tree[d-1][p].couple_state, vect(tree[d][i].couple_state))
            push!(tree[d-1][p].dual_state, vect(tree[d][i].dual_state))
        end
    end
end

query!(rootnode, tree)

# #Forward
# for d in 1:D
#     n = size(tree[d])
#     for i in 1:n
#         response!(tree[d][i])
#     end
# end

#Backward