function update_leaf!(node::linknode, query::Vector{Float64}; λ = λₙ)
    node.cost_func.q = query
    node.cost_func.λ = λ

    f = node.cost_func

    g = ProximalOperators.NormL1(node.cost_func.w)
    
    x0 = ones(node.nV)

    solution, iterations = node.solver(f = f, g = g, x0 = x0)

    node.prime[node.ID] = solution
end


function update_parent!(node::linknode, top_query::Vector{Float64}, query::Vector{Float64}; λ = λₙ)
    
    node.cost_func.q = (query+top_query)/2
    node.cost_func.λ = λ/2

    f = node.cost_func

    g = ProximalOperators.NormL1(node.cost_func.w)

    x0 = ones(node.nV)

    solution, iterations = node.solver(f = f, g = g, x0 = x0)

    index = 1
    for child in node.children
        node.prime[child.ID] = solution[index:index + child.nV - 1]
        index += child.nV
    end
end


function update_root!(node::linknode, query::Vector{Float64}; λ = λₙ)
    node.cost_func.q = query
    node.cost_func.λ = λ

    f = node.cost_func

    g = ProximalOperators.NormL1(node.cost_func.w)

    x0 = ones(node.nV)

    solution, iterations = node.solver(f = f, g = g, x0 = x0)

    index = 1
    for child in node.children
        node.prime[child.ID] = solution[index:index + child.nV - 1]
        index += child.nV
    end
end


function nestedADMM!(node::linknode, top_query = 0.; tol = tol, max_iter = max_iter, dict_result = nothing, traj_err = nothing, traj_res = nothing, traj_opt = nothing, traj_com = nothing)

    node.iteration += 1

    if node.children !== nothing
        for iteration in 1:max_iter 

            ter = Float64[]

            child_primes = Float64[] 
            for child in node.children
                cp = vect_prime(child) #take prime variables from children
                child_primes = append!(child_primes, cp) #already had in the last step. Thus, do not count in com.
            end
            qToPar  = child_primes - vect_dual(node) #com

            if node.parent === nothing #Root
                update_root!(node, qToPar)
            else
                update_parent!(node, top_query, qToPar)
                com_cost!(node.parent, top_query, 1) # Parent sent top_query to node
            end

            for child in node.children
                qToChi  = node.prime[child.ID] + node.dual[child.ID]
            
                nestedADMM!(child, qToChi, traj_com=traj_com)

                child_prime = vect_prime(child)
                res = node.prime[child.ID] - child_prime
                node.dual[child.ID]  += res
                append!(ter, res)

                com_cost!(child, child_prime, 1) #Child sent its prime variable to node
            end

            # Track metrics for root node AFTER all updates
            if node.parent === nothing && dict_result !== nothing && traj_err !== nothing && traj_res !== nothing && traj_opt !== nothing
                push!(traj_opt, total_cost(node))
                push!(traj_res, maximum(abs.(ter)))
                err = Float64[]
                get_err!(node, dict_result, err)
                push!(traj_err, sum(err))
                
                # Track total communication after this iteration
                if traj_com !== nothing
                    total, _ = tt_com_iter(node)
                    push!(traj_com, total["com"])
                end
            end

            if maximum(abs.(ter)) < tol
                if node.parent === nothing
                    println("nADMM converged after $iteration iterations in root")
                end
                
                break
            end
        end
    else
        com_cost!(node.parent, top_query, 1)
        update_leaf!(node, top_query)
    end
end
