#Vectorize prime dict of a node in the same order as its children
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

    g = ProximalOperators.NormL1(cost_func.w)
    return f, g
end

function proxAlg!(node::linknode; λ = λₕ)  

    nc        = 0
    q         = Float64[]
    x0        = vect_prime(node)
    
    if node.children === nothing #Leaf nodes
        q = node.parent.prime[node.ID] + node.parent.dual[node.ID] #query

        com_cost!(node.parent, q, 1) #Parent sent to node

        f, g = Proximal_Iteration(node.cost_func, q, λ)
    elseif node.parent === nothing #Root node
        nc = length(node.children)

        for i in 1:nc
            qc = vect_prime(node.children[i]) - node.dual[node.children[i].ID]
            q  = [q; qc]
        end

        f, g = Proximal_Iteration(node.cost_func, q, λ)
    else #Middle node
        nc = length(node.children)

        par  = node.parent.prime[node.ID] + node.parent.dual[node.ID]

        com_cost!(node.parent, par, 1) #Parent sent to node

        ch_pr = Float64[]
        for i in 1:nc
            qc = vect_prime(node.children[i])
            ch_pr = [ch_pr; qc]
        end
        

        q = [q; par - vect_dual(node) + ch_pr]
        f, g = Proximal_Iteration(node.cost_func, q/2, λ/2)
    end

    solution, iterations = node.solver(f = f, g = g, x0 = x0)

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


function hierarchicalADMM!(node::linknode, ter::Vector{Float64})

    node.iteration += 1
    
    #Update prime
    proxAlg!(node)

    if node.children !== nothing
        for child in node.children
            hierarchicalADMM!(child, ter)
            #Update dual
            child_prime = vect_prime(child)
            res = node.prime[child.ID] - child_prime
            node.dual[child.ID] += res

            # Child sent its prime var to node
            com_cost!(child, child_prime, 1)

            #Take the maximum residual error for stopping criteria
            push!(ter, maximum(abs.(res)))
        end
    end
end


function hADMM(root::linknode, dict_result::Dict; tol = tol, λ = λₕ, max_iter = max_iter)
    global stop_arr
    
    traj_err = Float64[]
    traj_res = Float64[]
    opt_value = Float64[]
    
    for iteration in 1:max_iter
        ter = Float64[]
        push!(opt_value, total_cost(root))
        hierarchicalADMM!(root, ter)

        err = Float64[]
        get_err!(root, dict_result, err)
        push!(traj_res, maximum(ter))
        push!(traj_err, sum(err))

        if maximum(ter) < tol
            println("hADMM converged after $iteration iterations in root")
            return traj_err, traj_res, opt_value
        end
    end

    return traj_err, traj_res, opt_value
end

# function forward_hierarchicalADMM!(root::linknode; max_nlayer = 10)
#     #Update prime 
#     VecLink = [root]

#     for _ in 1:max_nlayer #maximum layer is 10
#         if !isempty(VecLink)
#             newVecLink = linknode[]
#             for node in VecLink
#                 prox!(node)
#                 if node.children !== nothing
#                     for child in node.children
#                         push!(newVecLink, child)
#                     end
#                 end
#             end
#             VecLink = newVecLink
#         end
#     end
# end


# function backward_hierarchicalADMM!(root::linknode, ter::Vector{Float64};  max_nlayer = 10)
#     VecLink = [root]

#     for _ in 1:max_nlayer #maximum layer is 10
#         if !isempty(VecLink)
#             newVecLink = linknode[]
#             max_abs = 0
#             for node in VecLink
#                 if node.children !== nothing
#                     for child in node.children
#                         res = node.prime[child.ID] - vect_prime(child)
#                         node.dual[child.ID] = node.dual[child.ID] + res
#                         max_abs = (max_abs > maximum(abs.(res))) ? max_abs : maximum(abs.(res))
#                         push!(ter, max_abs)
#                         if child.children !== nothing
#                             push!(newVecLink, child)
#                         end
#                     end
#                 end
#             end
#             VecLink = newVecLink
#         end
#     end
# end