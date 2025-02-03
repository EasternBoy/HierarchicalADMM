using LinearAlgebra, Optim, JuMP, Plots

mutable struct node
    level::Int64
    parent::Int64
    children::Vector{Int64}
    local_state::Vector{Float64}
    cost::Float64
    couple_state::Vector{Vector{Float64}}
    dual_state::Vector{Vector{Float64}}
    opti

    function node(level::Int64, parent::Int64, children::Vector{Int64}, local_state::Vector{Float64}, cost::Float64)
        obj = new(level, parent, children, local_state, cost)
        if length(children) == 0
            obj.couple_state = [local_state]
        else
            obj.couple_state = Vector{Float64}[]
            dual_state = Vector{Float64}[]
        end
        
        obj.opti = JuMP.Model(Optim.Optimizer)
        return obj
    end
end

function query!(nd::node, tree::Vector{Vector{node}})
    return 0
end

function response!(parent::node, tree::Vector{Vector{node}})
    return 0
end

function prox!(nd::node, tree::Vector{node})
    nd.opti = JuMP.Model(Optim.Optimizer)
    set_optimizer_attribute(nd.opti, "method", BFGS())
    set_silent(nd.opti)

    


    λ = 1.
    n = length(nd.children)
    l = [length(nd.couple_state[i]) for i in 1:n]
    c = nd.cost

    if node.level == 0
        λ = lam
        @variable(model, x[i = 1:length(nd.children), 1:l[i]])

        q = [nd.dual_state[i] + tree[1][i].couple_state for i in 1:n]

        J = sum(dot(x[i] .- c, x[i] .- c) for i in 1:n) + 1/(2λ)*sum(dot(x[i] - q[i], x[i] - q[i]) for i in 1:n)
    elseif length(node.children) == 0
        @variable(model, x[i:length(nd.couple_state)])
    else
        @variable(model, x[i = 1:length(nd.children), 1:l[i]])
    end



    @variable(robo.opti, x[1:n])
end

function vect(v::Vector{Vector{Float64}})
    res = Float64[]
    for i in 1:length(v)
        res = [res; v[i]]
    end
    return res
end