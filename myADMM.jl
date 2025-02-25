include("TreeGeneration.jl")

#Vectorize prime dict of a node in the same order as its children
function vectp(node::linknode)
    result = Float64[]
    if node.children === nothing #leaf: only element in its prime dict
        result = node.prime[node.ID]
    else #middle nodes: 
        for i in 1:length(node.children)
            result = [result; node.prime[node.children[i].ID]]
        end
    end
    return result
end

function prox!(node::linknode; λ = 0.1)    
    if node.children === nothing #Leaf nodes
        q = node.parent.prime[node.ID] + node.parent.dual[node.ID] #query

        opti = JuMP.Model(Optim.Optimizer)
        @variable(opti, x[1:node.nV])
        c = parse(Float64, node.ID)
        J = node.costfunction(x, c) + 1/(2λ)*(x - q)'*(x - q)
    elseif node.parent === nothing #Root node
        nc = length(node.children)
        q  = [vectp(node.children[i]) - node.dual[node.children[i].ID] for i in 1:nc]

        opti = JuMP.Model(Optim.Optimizer)
 
        l = [node.children[i].nV for i in 1:nc]
        @variable(opti, x[i = 1:nc, 1:l[i]])

        c = parse(Float64, node.ID)
        J = sum(node.costfunction([x[i,j] for j in l[i]], c) for i in 1:nc) + 1/(2λ)*sum(sum((x[i,j] - q[i][j])^2 for j in 1:l[i]) for i in 1:nc)
    else #Middle node
        nc = length(node.children)
        q  = node.parent.prime[node.ID] + node.parent.dual[node.ID]

        v = Vector{Vector{Float64}}(undef, nc)
        l = [node.children[i].nV for i in 1:nc]
        k = 1
        for i in 1:nc
            v[i] = q[k:(k + l[i] - 1)]
            k = k + l[i]
        end

        res = v - [node.dual[node.children[i].ID] for i in 1:nc] + [vectp(node.children[i]) for i in 1:nc]

        opti = JuMP.Model(Optim.Optimizer)
        
        @variable(opti, x[i = 1:nc, 1:l[i]])
        c = parse(Float64, node.ID)

        J = sum(node.costfunction([x[i,j] for j in l[i]], c) for i in 1:nc) + (1/λ)*sum(sum((x[i,j] - res[i][j]/2)^2  for j in 1:l[i]) for i in 1:nc)
    end

    @objective(opti, Min, J)
    JuMP.optimize!(opti)
    x = JuMP.value.(x)

    #Update prime variable
    if node.children === nothing #leaf node has only one
        node.prime[node.ID] = x
    else
        for i in 1:nc
            ID = node.children[i].ID
            node.prime[ID] = [x[i,j] for j in 1:node.children[i].nV]
        end
    end
end

function hierarchicalADMM!(node::linknode)
    ter = 0
    #Update prime
    prox!(node)
    if node.children !== nothing
        for child in node.children
            temp_ter = hierarchicalADMM!(child)
            #Update dual
            res = child.parent.prime[child.ID] -  vectp(child)
            child.parent.dual[child.ID] += res
            #Take the maximum residual error for stopping criteria
            ter = (temp_ter > maximum(abs.(res))) ? temp_ter : maximum(abs.(res))
        end
    else
        #Update dual
        res = node.parent.prime[node.ID] -  vectp(node)
        node.parent.dual[node.ID] += res
        #Take the maximum residual error for stopping criteria
        ter = (ter > maximum(abs.(res))) ? ter : maximum(abs.(res))
    end

    return ter
end