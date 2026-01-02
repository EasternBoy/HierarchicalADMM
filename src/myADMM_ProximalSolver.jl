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

function vect_dual(node::linknode)
    result = Float64[]
    for child in node.children
        result = [result; node.dual[child.ID]]
    end
    return result
end

function Proximal_Iteration(p::Float64, q::Union{Float64, Vector{Float64}}, λ::Float64)
    f = ProximalAlgorithms.AutoDifferentiable(
        x -> 1/(2λ) * dot(x .- q, x .- q) + cost_func(x, p),
        AutoZygote()
    )
    g = ProximalOperators.IndBox(1e-6, Inf)
    return f, g
end

function proxAlg!(node::linknode; λ = 0.01)  
    global tot_com

    nc = 0
    p  = parse(Float64, node.ID)
    q  = Float64[]
    x0 = vectp(node)

    
    if node.children === nothing #Leaf nodes
        q = node.parent.prime[node.ID] + node.parent.dual[node.ID] #query

        tot_com += 1    # Received prime + dual from parent

        f, g = Proximal_Iteration(p, q, λ)
    elseif node.parent === nothing #Root node
        nc = length(node.children)

        for i in 1:nc
            q = [q; vectp(node.children[i]) - node.dual[node.children[i].ID]]
        end

        tot_com += nc   #Received prime from all children

        f, g = Proximal_Iteration(p, q, λ)
    else #Middle node
        nc = length(node.children)

        par  = node.parent.prime[node.ID] + node.parent.dual[node.ID]

        tot_com += 1    #Received prime + dual from parent

        ch_pr = Float64[]
        for i in 1:nc
            ch_pr = [ch_pr; vectp(node.children[i])]
        end
        # Received prime from all children
        tot_com += nc

        q = [q; par - vect_dual(node) + ch_pr]
        f, g = Proximal_Iteration(p, q/2, λ/2)
    end

    solution, iterations = node.prox(f = f, g = g, x0 = x0)

    #Update prime variable
    if node.children === nothing #leaf node has only one
        node.prime[node.ID] = solution
    else
        dc = [node.children[i].nV for i in 1:nc]
        index = 1
        for i in 1:nc
            ID = node.children[i].ID
            node.prime[ID] = solution[index:(index + dc[i]-1)]
            index += dc[i]
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
    proxAlg!(node)

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