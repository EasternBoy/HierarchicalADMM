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
    x0        = node.prime #Initial value is the prime variable of the node
    
    if node.children === nothing #Leaf nodes
        q = node.parent.prime + node.parent.dual[node.ID] #query

        com_cost!(node.parent, q, 1) #Parent sent to node

        f, g = Proximal_Iteration(node.cost_func, q, λ)
    elseif node.parent === nothing #Root node
        nc = length(node.children)

        q = zeros(node.nV)

        for i in 1:nc
            childID = node.children[i].ID
            qc = node.children[i].prime  - node.dual[childID]
            q  += qc
        end

        f, g = Proximal_Iteration(node.cost_func, q/nc, λ/nc)
    else #Parent node
        nc = length(node.children)

        p  = node.parent.prime + node.parent.dual[node.ID]

        com_cost!(node.parent, p, 1) #Parent sent to node

        for i in 1:nc
            p += node.children[i].prime - node.dual[node.children[i].ID]
        end
        
        f, g = Proximal_Iteration(node.cost_func, p/(nc+1), λ/(nc+1))
    end

    solution, iterations = node.solver(f = f, g = g, x0 = x0)

    #Update prime variable

    node.prime = solution
end


function hierarchicalADMM!(node::linknode, ter::Vector{Float64})

    node.iteration += 1
    
    #Update prime
    proxAlg!(node)

    if node.children !== nothing
        for child in node.children
            hierarchicalADMM!(child, ter)
            #Update dual
            res = node.prime - child.prime
            node.dual[child.ID] += res

            # Child sent its prime var to node
            com_cost!(child, child.prime, 1)

            #Take the maximum residual error for stopping criteria
            push!(ter, maximum(abs.(res)))
        end
    end
end


function hADMM(root::linknode, opt_sol; tol = tol, λ = λₕ, max_iter = max_iter)
    global stop_arr
    
    traj_err = Float64[]
    traj_res = Float64[]
    opt_value = Float64[]
    
    for iteration in 1:max_iter
        ter = Float64[]
        push!(opt_value, total_cost(root))
        hierarchicalADMM!(root, ter)

        err = Float64[]
        get_err!(root, opt_sol, err)
        push!(traj_res, maximum(ter))
        push!(traj_err, sum(err))

        if maximum(ter) < tol
            # println("hADMM converged after $iteration iterations in root")
            return traj_err, traj_res, opt_value, iteration
        end
    end

    return traj_err, traj_res, opt_value
end

function get_err!(node::linknode, opt_sol, err::Vector{Float64})
    if node.children === nothing
        push!(err, maximum(abs.(node.prime - opt_sol)))
    else
        for child in node.children
            get_err!(child, opt_sol, err)
        end
    end
end