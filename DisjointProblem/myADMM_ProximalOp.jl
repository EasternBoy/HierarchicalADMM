include("TreeGeneration.jl")
#Vectorize prime dict of a node in the same order as its children
function vectp(node::linknode)
    result = Float64[]
    if node.children === nothing #leaf: only element in its prime dict
        result = node.prime[node.ID]
    else #middle nodes: 
        for child in node.children
            result = [result; node.prime[child.ID]]
        end
    end
    return result
end

function prox!(node::linknode; λ = 0.01)  
    global tot_com

    opti = JuMP.Model(Optim.Optimizer)
    cst  = parse(Float64, node.ID)
    set_optimizer_attribute(opti, "method", BFGS())
    set_silent(opti)

    
    if node.children === nothing #Leaf nodes
        q = node.parent.prime[node.ID] + node.parent.dual[node.ID] #query
        # Received prime + dual from parent
        tot_com += 1

        @variable(opti, x[1:node.nV])

        J = costfunction(x, cst) + 1/(2λ)*(x - q)'*(x - q)

    elseif node.parent === nothing #Root node
        nc = length(node.children)
        dc  = [node.children[i].nV for i in 1:nc]

        q  = [vectp(node.children[i]) - node.dual[node.children[i].ID] for i in 1:nc]
        # Received prime from all children
        tot_com += nc
 
        @variable(opti, x[i = 1:nc, 1:dc[i]])

        J = costfunction(x, cst, nc) + 1/(2λ)*sum(sum((x[i,j] - q[i][j])^2 for j in 1:dc[i]) for i in 1:nc)
    else #Middle node
        nc = length(node.children)
        dc = [node.children[i].nV for i in 1:nc]

        par  = node.parent.prime[node.ID] + node.parent.dual[node.ID]
        # Received prime + dual from parent
        tot_com += 1

        par_vec = Vector{Vector{Float64}}(undef, nc)
        index = 1
        for i in 1:nc
            par_vec[i] = par[index:(index + dc[i] - 1)]
            index = index + dc[i]
        end

        res = par_vec - [node.dual[node.children[i].ID] for i in 1:nc] + [vectp(node.children[i]) for i in 1:nc]
        # Received prime from all children
        tot_com += nc
        
        @variable(opti, x[i = 1:nc, 1:dc[i]])

        J = costfunction(x, cst, nc) + (1/λ)*sum(sum((x[i,j] - 1/2*res[i][j])^2  for j in 1:dc[i]) for i in 1:nc)
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

function forward_hierarchicalADMM!(root::linknode; max_nlayer = 10)
    #Update prime 
    VecLink = [root]

    for _ in 1:max_nlayer #maximum layer is 10
        if !isempty(VecLink)
            newVecLink = linknode[]
            for node in VecLink
                prox!(node)
                if node.children !== nothing
                    for child in node.children
                        push!(newVecLink, child)
                    end
                end
            end
            VecLink = newVecLink
        end
    end
end


function backward_hierarchicalADMM!(root::linknode, ter::Vector{Float64};  max_nlayer = 10)
    VecLink = [root]

    for _ in 1:max_nlayer #maximum layer is 10
        if !isempty(VecLink)
            newVecLink = linknode[]
            max_abs = 0
            for node in VecLink
                if node.children !== nothing
                    for child in node.children
                        res = node.prime[child.ID] - vectp(child)
                        node.dual[child.ID] = node.dual[child.ID] + res
                        max_abs = (max_abs > maximum(abs.(res))) ? max_abs : maximum(abs.(res))
                        push!(ter, max_abs)
                        if child.children !== nothing
                            push!(newVecLink, child)
                        end
                    end
                end
            end
            VecLink = newVecLink
        end
    end
end


function hierarchicalADMM!(node::linknode, ter::Vector{Float64})
    global tot_com

    #Update prime
    prox!(node)

    if node.children !== nothing
        for child in node.children
            hierarchicalADMM!(child, ter)
            #Update dual
            res = child.parent.prime[child.ID] -  vectp(child)
            child.parent.dual[child.ID] += res
            # Receive a prime var of one child for updating a part of dual
            tot_com += 1
            #Take the maximum residual error for stopping criteria
            push!(ter, maximum(abs.(res)))
        end
    end
end