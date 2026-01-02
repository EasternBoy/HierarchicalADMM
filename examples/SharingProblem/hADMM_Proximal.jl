include("TreeGeneration.jl")

using LinearAlgebra, JuMP, Optim
using Zygote
using ProximalAlgorithms
using ProximalOperators
using DifferentiationInterface: AutoZygote


function ProximalAlgorithms.value_and_gradient(cost_func::CostFunc, x)
    prm  = cost_func.para
    q = cost_func.q 
    λ = cost_func.λ

    val  = cost_func.val(x; para = prm)  + 1/(2λ)*dot(x - q, x - q)
    grad = cost_func.grad(x; para = prm) + 1/λ*(x - q)
    return val, grad
end


function Proximal_Iteration(cost_func::CostFunc, q::Union{Float64, Vector{Float64}}, λ::Float64)
    cost_func.q = q
    cost_func.λ = λ
    f = cost_func
    g = ProximalOperators.IndBox(0., Inf)
    return f, g
end


function proxAlg!(node::linknode; λ = λₕ)  
    nc        = 0
    q         = Float64[]
    x0        = vect_prime(node)

    if node.children === nothing #Leaf nodes
        q = node.parent.prime[node.ID] + node.parent.dual[node.ID] #query

        com_cost!(node.parent,q, 1)

        f, g = Proximal_Iteration(node.cost_func, q, λ)
    elseif node.parent === nothing #Root node
        nc = length(node.children)

        for i in 1:nc
            qc = vect_prime(node.children[i]) - node.dual[node.children[i].ID]
            q  = [q; vect_prime(node.children[i]) - node.dual[node.children[i].ID]]
        end

        f, g = Proximal_Iteration(node.cost_func, q, λ)
    else #Middle node
        nc = length(node.children)

        par  = node.parent.prime[node.ID] + node.parent.dual[node.ID]

        com_cost!(node.parent, par, 1) #Received prime + dual from parent

        ch_pr = Float64[]
        for i in 1:nc
            qc = vect_prime(node.children[i])
            ch_pr = [ch_pr; qc]
            # com_cost!(node.children[i], qc, 1) #Received prime from all child i
        end
    
        q = [q; par - vect_dual(node) + ch_pr]
        f, g = Proximal_Iteration(node.cost_func, q/2, λ/2)
    end

    solution, iterations = node.solver(x0 = x0, f = f, g = g)

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


function HADMM_Prox!(node::linknode, ter::Vector{Float64})

    node.iteration += 1
    #Update prime
    proxAlg!(node)

    if node.children !== nothing
        for child in node.children
            HADMM_Prox!(child, ter)

            #Update dual
            child_prime = vect_prime(child)
            res = child.parent.prime[child.ID] -  child_prime
            child.parent.dual[child.ID] += res

            # Receive a prime variable of one child for updating dual variable
            com_cost!(child, child_prime, 1)

            # Take the maximum residual error for stopping criteria
            push!(ter, maximum(abs.(res)))
        end
    end
end