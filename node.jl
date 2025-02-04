using LinearAlgebra, Optim, JuMP, Plots

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

function query!(nd::node, tree::Vector{Vector{node}})
    return 0
end

function response!(parent::node, tree::Vector{Vector{node}})
    return 0
end

function prox!(nd::node, tTree::Tuple{node, Vector{Vector{node}}})

    root  = tTree[1]
    tree  = tTree[2]

    nd.opti = JuMP.Model(Optim.Optimizer)
    set_optimizer_attribute(nd.opti, "method", BFGS())
    set_silent(nd.opti)

    λ = 1.
    d = nd.level
    p = nd.parent
    n = length(nd.children)
    l = [length(nd.couple_state[i]) for i in 1:n]
    c = nd.cost

    if d == 0 #Root node
        @variable(nd.opti, x[i = 1:n, 1:l[i]])

        q = [nd.dual_state[i] + vect(tree[1][i].couple_state) for i in 1:n]

        J = sum(dot(x[i,:] .- c, x[i,:] .- c) for i in 1:n) + 1/(2λ)*sum(sum((x[i,j] - q[i][j])^2 for j in 1:l[i]) for i in 1:n)
    else
        parent = (d==1) ? root : tree[d-1][p]
        if n == 0 #Leaf nodes
            @variable(nd.opti, x[1:length(nd.couple_state)])

            i = indexin(nd.index, parent.children)[1]

            q = parent.couple_state[i] - parent.dual_state[i]

            J = dot(x .- c, x .- c) + 1/(2λ)*dot(x - q, x - q)
        else #Middle node
            @variable(nd.opti, x[i = 1:length(nd.children), 1:l[i]])

            i = indexin(nd.index, parent.children)[1]

            q = parent.couple_state[i] - parent.dual_state[i]

            u = nd.dual_state

            k = 1
            for i in 1:n
                u[i] = u[i] + q[k:(k+l[i]-1)]
                k = k+l[i]
            end

            xchild = [vect(tree[d+1][j].couple_state) for j in nd.children]

            res = xchild + u

            J = sum(dot(x[i,:] .- c, x[i,:] .- c) for i in 1:n) + 1/λ*sum(sum((x[i,j] - res[i][j])^2 for j in 1:l[i]) for i in 1:n)
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