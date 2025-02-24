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

function prox!(node::linknode; λ = 1.0)
    println(node.ID)
    
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

function hierarchicalADMM!(node::linknode, depth::Int64; max_iter = 20, tol = 1e-2)

    termination = 0 #Reset termination

    for step in 1:max_iter
        prox!(node)
        temp_vec_node = node.children

        for _ in 1:depth 
            new_vec_node = linknode[]
            for subnode in temp_vec_node
                # Forward
                prox!(subnode)
                # Forward
                res = subnode.parent.prime[subnode.ID] -  vect(subnode.prime)
                subnode.parent.dual[subnode.ID] += res

                if termination < norm(res) #take maximum res as stopping criteria
                    termination = norm(res)
                end

                if subnode.children !== nothing
                    for child in subnode.children
                        new_vec_node = [new_vec_node; child]
                    end
                end
            end
            temp_vec_node = new_vec_node
        end

        if termination < tol
            println("Stop at step $step")
            return step
        end
        termination = 0 #Reset termination
    end
end


# root=linknode(string(countID+=1))
nN = 15
nD = 5
# topo_gen!(root, nN, nD)
# assign_var!(root)

root = load("myfile.jld2", "root")
reset_var!(root)

hierarchicalADMM!(root, nN)

print_tree(root)