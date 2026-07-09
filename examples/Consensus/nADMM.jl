function update_leaf!(node::linknode, query::Vector{Float64}; λ = λₙ)
    node.cost_func.q = query
    node.cost_func.λ = λ

    f = node.cost_func

    # g = ProximalOperators.NormL1(node.cost_func.w)
    g = ProximalOperators.CubeNormL2(node.cost_func.w)
    
    x0 = ones(node.nV)

    solution, iterations = node.solver(f = f, g = g, x0 = x0)

    node.prime[node.ID] = solution
end


function update_parent!(node::linknode, top_query::Vector{Float64}, query::Vector{Float64}; λ = λₙ)
    
    node.cost_func.q = (query+top_query)/2
    node.cost_func.λ = λ/2

    f = node.cost_func

    # g = ProximalOperators.NormL1(node.cost_func.w)
    g = ProximalOperators.CubeNormL2(node.cost_func.w)


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

    # g = ProximalOperators.NormL1(node.cost_func.w)
    g = ProximalOperators.CubeNormL2(node.cost_func.w)


    x0 = ones(node.nV)

    solution, iterations = node.solver(f = f, g = g, x0 = x0)

    index = 1
    for child in node.children
        node.prime[child.ID] = solution[index:index + child.nV - 1]
        index += child.nV
    end
end


function nestedADMM!(node::linknode, top_query = 0.; tol = tol, max_iter = max_iter, λ = λₙ)

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
                update_root!(node, qToPar; λ = λ)
            else
                update_parent!(node, top_query, qToPar; λ = λ)
                com_cost!(node.parent, top_query, 1) # Parent sent top_query to node
            end

            for child in node.children
                qToChi  = node.prime[child.ID] + node.dual[child.ID]
                child_prime_old = copy(vect_prime(child))
            
                nestedADMM!(child, qToChi; tol = tol, max_iter = max_iter, λ = λ)

                child_prime = vect_prime(child)
                prime_res = node.prime[child.ID] - child_prime
                dual_res = (child_prime - child_prime_old) / λ
                node.dual[child.ID] += prime_res
                push!(ter, max(norm(prime_res, Inf), norm(dual_res, Inf)))

                com_cost!(child, child_prime, 1) #Child sent its prime variable to node
            end

            if maximum(ter) < tol
                if node.parent === nothing
                    println("nADMM converged after $iteration iterations in root")
                end
                
                break
            end
        end
    else
        com_cost!(node.parent, top_query, 1)
        update_leaf!(node, top_query; λ = λ)
    end
end
