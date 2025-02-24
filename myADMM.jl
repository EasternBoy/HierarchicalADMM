include("TreeGeneration.jl")

function vect(dict::Dict)
    result = Float64[]
    for (x,y) in dict
        result = [result; y]
    end
    return result
end

function vect(v::Vector{Vector{Float64}})
    res = Float64[]
    for i in 1:length(v)
        res = [res; v[i]]
    end
    return res
end

function prox!(node::linknode; λ = 0.1)

    
    if node.children === nothing #Leaf nodes
        opti = JuMP.Model(Optim.Optimizer)

        var = @variable(opti, [1:node.nV])

        q = node.parent.prime[node.ID] + node.parent.dual[node.ID] #query

        J = node.costfunction(var, parse(Float64, node.ID)) + 1/(2λ)*dot(var - q, var - q)
    elseif node.parent === nothing #Root node
        nc   = length(node.children)
        opti = JuMP.Model(Optim.Optimizer)
        var  = Vector{Vector{VariableRef}}(undef, nc)

        for i in 1:nc
            var[i] =  @variable(opti, [1:node.children[i].nV])
        end

        q = [vect(node.children[i].prime) - node.dual[node.children[i].ID] for i in 1:nc]

        J = sum(node.costfunction(var[i], parse(Float64, node.ID))  + 1/(2λ)*(var[i] - q[i])'*(var[i] - q[i])  for i in 1:nc)
    else
        nc   = length(node.children)
        opti = JuMP.Model(Optim.Optimizer)
        var  = Vector{Vector{VariableRef}}(undef, nc)

        res  = node.parent.prime[node.ID] + node.parent.dual[node.ID] #query
        res  = res - vect(node.dual) + vect([vect(node.children[i].prime) for i in 1:nc])

        v = []
        for i in 1:nc
            var[i] =  @variable(opti, [1:node.children[i].nV])
            v = [v; var[i]]
        end

        J = sum(node.costfunction(var[i], parse(Float64, node.ID)) for i in 1:nc) + 1/λ*(v - res/2)'*(v - res/2)   
    end

    @objective(opti, Min, J)
    JuMP.optimize!(opti)

    if node.children === nothing #leaf node has only one
        node.prime[node.ID] = JuMP.value.(var)
    else
        for i in 1:nc
            ID = node.children[i].ID
            node.prime[ID] = JuMP.value.(var[i])
        end
    end
end

function termination(node::linknode)
    return false
end

function hierarchicalADMM!(node::linknode; max_iter = 1)
    for k in 1:max_iter
        prox!(node)
        for child in node.children
            prox!(child)
        end
    end
end

hierarchicalADMM!(root)