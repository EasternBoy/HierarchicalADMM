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


function nestedADMM!(node::linknode, top_query = 0.; tol = tol, max_iter = max_iter, last_dual = nothing)

    if last_dual === nothing
        last_dual = Dict{String, Float64}()
    end

    node.iteration += 1

    if node.children !== nothing
        for iteration in 1:max_iter 

            ter = Float64[]
            prev_primes = Dict{String, Vector{Float64}}()
            for child in node.children
                prev_primes[child.ID] = copy(node.prime[child.ID])
            end

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

            dual_res = 0.0
            for child in node.children
                dual_res = max(dual_res, maximum(abs.(node.prime[child.ID] - prev_primes[child.ID])))
                qToChi  = node.prime[child.ID] + node.dual[child.ID]
            
                nestedADMM!(child, qToChi; tol = tol, max_iter = max_iter, last_dual = last_dual)

                child_prime = vect_prime(child)
                res = node.prime[child.ID] - child_prime
                node.dual[child.ID]  += res
                append!(ter, res)

                com_cost!(child, child_prime, 1) #Child sent its prime variable to node
            end

            last_dual[node.ID] = dual_res

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
        last_dual[node.ID] = 0.0
    end

    return last_dual
end
