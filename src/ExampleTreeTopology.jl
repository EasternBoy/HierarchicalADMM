push!(LOAD_PATH, ".")


import Pkg
using Pkg
Pkg.activate(@__DIR__)

using LinearAlgebra, Optim, JuMP

mutable struct node
    level::Int64
    index::Int64
    parent::Int64
    children::Vector{Int64}
    local_state::Vector{Float64}
    cost::Float64
    couple_state
    dual_state
    opti

    function node(level::Int64, index::Int64, parent::Int64, children::Vector{Int64}, local_state::Vector{Float64}, cost::Float64)
        obj = new(level, index, parent, children, local_state, cost)
        if length(children) == 0
            obj.couple_state = local_state
            obj.dual_state   = local_state
        else
            obj.couple_state = Vector{Float64}[]
            obj.dual_state   = Vector{Float64}[]
        end
        
        obj.opti = JuMP.Model(Optim.Optimizer)
        return obj
    end
end


function vect(v::Vector{Vector{Float64}})
    res = Float64[]
    for i in 1:length(v)
        res = [res; v[i]]
    end
    return res
end

function vect(v::Vector{Float64})
    return v
end

function prox!(nd::node, tTree::Tuple{node, Vector{Vector{node}}})

    root  = tTree[1]
    tree  = tTree[2]

    nd.opti = JuMP.Model(Optim.Optimizer)
    set_optimizer_attribute(nd.opti, "method", BFGS())
    set_silent(nd.opti)

    λ = 0.1
    d = nd.level
    p = nd.parent
    n = length(nd.children)
    l = [length(nd.couple_state[i]) for i in 1:n]
    c = nd.cost

    if d == 0 #Root node
        @variable(nd.opti, x[i = 1:n, 1:l[i]])

        q = [vect(tree[1][i].couple_state) - nd.dual_state[i] for i in 1:n]

        J = sum(dot(x[i,:] .- c, x[i,:] .- c) for i in 1:n)^2 + 1/(2λ)*sum(sum((x[i,j] - q[i][j])^2 for j in 1:l[i]) for i in 1:n)
    else
        parent = (d==1) ? root : tree[d-1][p]
        if n == 0 #Leaf nodes
            @variable(nd.opti, x[1:length(nd.couple_state)])

            ic = indexin(nd.index, parent.children)[1]

            q = parent.couple_state[ic] + parent.dual_state[ic] #query

            J = dot(x .- c, x .- c)^2 + 1/(2λ)*dot(x - q, x - q)
        else #Middle node
            @variable(nd.opti, x[i = 1:length(nd.children), 1:l[i]])

            ic = indexin(nd.index, parent.children)[1]

            q  = parent.couple_state[ic] + parent.dual_state[ic] #query

            qv = Vector{Vector{Float64}}(undef, n)

            k = 1
            for i in 1:n
                qv[i] = q[k:(k + l[i] - 1)]
                k = k + l[i]
            end

            xchild = [vect(tree[d+1][i].couple_state) for i in nd.children]

            res = xchild - nd.dual_state + qv

            J = sum(dot(x[i,:] .- c, x[i,:] .- c) for i in 1:n)^2 + (1/λ)*sum(sum((x[i,j] - 1/2*res[i][j])^2 for j in 1:l[i]) for i in 1:n)
        end
    end
    
    @objective(nd.opti, Min, J)
    JuMP.optimize!(nd.opti)
    x = JuMP.value.(x)
    if n == 0
        nd.couple_state = x
    else
        for i in 1:n
            nd.couple_state[i] = [x[i,j] for j in 1:l[i]]
        end
    end

    return nd.couple_state
end


nlocal = 2

D = 4
root     = node(0, 1,-1, [1,2], zeros(nlocal), 0.)

node11   = node(1, 1, 0, [1,2,3], zeros(nlocal), 1.1)
node12   = node(1, 2, 0, Int64[], zeros(nlocal), 1.2)

node21   = node(2, 1, 1, [1,2,3], zeros(nlocal), 2.1)
node22   = node(2, 2, 1, Int64[], zeros(nlocal), 2.2)
node23   = node(2, 3, 1, Int64[], zeros(nlocal), 2.3)

node31   = node(3, 1, 1, Int64[], zeros(nlocal), 3.1)
node32   = node(3, 2, 1, Int64[], zeros(nlocal), 3.2)
node33   = node(3, 3, 1, Int64[1,2,3], zeros(nlocal), 3.3)

node41   = node(4, 1, 3, Int64[], zeros(nlocal), 3.1)
node42   = node(4, 2, 3, Int64[], zeros(nlocal), 3.2)
node43   = node(4, 3, 3, Int64[], zeros(nlocal), 3.3)



tree = [[node11, node12],
        [node21, node22, node23], 
        [node31, node32, node33],
        [node41, node42, node43]];

for d in D:-1:1
    for i in 1:length(tree[d])
        if d == 1
            push!(root.couple_state, vect(tree[d][i].couple_state))
            push!(root.dual_state,   vect(tree[d][i].dual_state))
        else
            p = tree[d][i].parent
            push!(tree[d-1][p].couple_state, vect(tree[d][i].couple_state))
            push!(tree[d-1][p].dual_state,   vect(tree[d][i].dual_state))
        end
    end
end

tTree = (root, tree)


max_iter = 200
tol = 1e-6
for k = 1:max_iter
    println("Step $k")
    #Forward
    prox!(root, tTree)
    for d in 1:D
        for i in 1:length(tree[d])
            prox!(tree[d][i], tTree)
        end
    end

    #Backward
    max_res = 0
    for d in D:-1:1
        for i in 1:length(tree[d])    
            if d == 1
                res = root.couple_state[i] - vect(tree[d][i].couple_state)
                root.dual_state[i] = root.dual_state[i] + res
                # println(round(norm(res), digits = 3))
            else
                p   = tree[d][i].parent
                ic  = indexin(tree[d][i].index, tree[d-1][p].children)[1]
                res = tree[d-1][p].couple_state[ic] - vect(tree[d][i].couple_state)
                max_res = (max_res > maximum(abs.(res))) ? max_res : maximum(abs.(res))
                tree[d-1][p].dual_state[ic]  = tree[d-1][p].dual_state[ic] + res
                # println(round(norm(res), digits = 3))
            end
        end
    end
    if max_res < tol
        break
    end
end


opti1 = JuMP.Model(Optim.Optimizer)

@variable(opti1, x12[1:2])
@variable(opti1, x22[1:2])
@variable(opti1, x23[1:2])
@variable(opti1, x31[1:2])
@variable(opti1, x32[1:2])
@variable(opti1, x41[1:2])
@variable(opti1, x42[1:2])
@variable(opti1, x43[1:2])



J = dot(x12 .- tree[1][2].cost, x12 .- tree[1][2].cost)^2 + #Leaf nodes
    dot(x22 .- tree[2][2].cost, x22 .- tree[2][2].cost)^2 + 
    dot(x23 .- tree[2][3].cost, x23 .- tree[2][3].cost)^2 + 
    dot(x31 .- tree[3][1].cost, x31 .- tree[3][1].cost)^2 +
    dot(x32 .- tree[3][2].cost, x32 .- tree[3][2].cost)^2 +
    dot(x41 .- tree[4][1].cost, x41 .- tree[4][1].cost)^2 + 
    dot(x42 .- tree[4][2].cost, x42 .- tree[4][2].cost)^2 + 
    dot(x43 .- tree[4][3].cost, x43 .- tree[4][3].cost)^2 +
    dot([x41;x42;x43] .- tree[3][3].cost, [x41;x42;x43] .- tree[3][3].cost)^2 +
    dot([x31;x32;x41;x42;x43] .- tree[2][1].cost, [x31;x32;x41;x42;x43] .- tree[2][1].cost)^2 +
    dot([x22;x23;x31;x32;x41;x42;x43] .- tree[1][1].cost, [x22;x23;x31;x32;x41;x42;x43] .- tree[1][1].cost)^2 + 
    dot([x12;x22;x23;x31;x32;x41;x42;x43] .- root.cost, [x12;x22;x23;x31;x32;x41;x42;x43] .- root.cost)^2 #root

@objective(opti1, Min, J)
@time JuMP.optimize!(opti1)

println("Differences between global solution and iteratively solved solution")
println(JuMP.value.(x12) - tree[1][2].couple_state)
println(JuMP.value.(x22) - tree[2][2].couple_state)
println(JuMP.value.(x23) - tree[2][3].couple_state)
println(JuMP.value.(x31) - tree[3][1].couple_state)
println(JuMP.value.(x32) - tree[3][2].couple_state)
println(JuMP.value.(x41) - tree[4][1].couple_state)
println(JuMP.value.(x42) - tree[4][2].couple_state)
println(JuMP.value.(x43) - tree[4][3].couple_state)