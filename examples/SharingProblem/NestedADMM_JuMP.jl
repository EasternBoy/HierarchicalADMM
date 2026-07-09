function update_leaf!(node::linknode, query::Vector{Float64}; λ = λₙ)
    opti = JuMP.Model(Ipopt.Optimizer)
    set_silent(opti)

    x0 = ones(node.nV)
    @variable(opti, x[i = 1:node.nV], base_name = string("x", node.ID), start = x0[i])
    @constraint(opti, x .>= 1e-5)

    para = node.cost_func.para

    @variable(opti, sl[i = 1:node.nV] .>= 0, base_name = string("sl", node.ID))
    @constraint(opti, sl .>= 1 ./ x .- 1 / para[1])

    @objective(opti, Min, sum(sl) + 1 / (2λ) * dot(x - query, x - query))
    JuMP.optimize!(opti)

    node.prime[node.ID] = JuMP.value.(x)
end


function update_parent!(node::linknode, top_query::Vector{Float64}, query::Vector{Float64}; λ = λₙ)
    opti = JuMP.Model(Ipopt.Optimizer)
    set_silent(opti)

    x0 = ones(node.nV)
    @variable(opti, x[i = 1:node.nV], base_name = string("x", node.ID), start = x0[i])
    @constraint(opti, x .>= 1e-5)

    para = node.cost_func.para
    η  = para[1]
    τ  = para[2]
    nc = para[3]
    horizon = Int64(node.nV / nc)

    @variable(opti, t[i = 1:horizon] .>= 0, base_name = string("t", node.ID))
    for i in 1:horizon
        @constraint(opti, t[i] >= sum([x[(j - 1) * horizon + i] for j in 1:nc]) - τ)
    end

    q = (top_query + query) / 2
    @objective(opti, Min, η * sum(t) + 1 / λ * dot(x - q, x - q))
    JuMP.optimize!(opti)

    solution = JuMP.value.(x)

    index = 1
    for child in node.children
        node.prime[child.ID] = solution[index:index + child.nV - 1]
        index += child.nV
    end
end


function update_root!(node::linknode, query::Vector{Float64}; λ = λₙ)
    opti = JuMP.Model(Ipopt.Optimizer)
    set_silent(opti)

    x0 = ones(node.nV)
    @variable(opti, x[i = 1:node.nV], base_name = string("x", node.ID), start = x0[i])
    @constraint(opti, x .>= 1e-5)

    func = node.cost_func.val
    para = node.cost_func.para

    @objective(opti, Min, func(x; para = para) + 1 / (2λ) * dot(x - query, x - query))
    JuMP.optimize!(opti)

    solution = JuMP.value.(x)

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
                child_primes = [child_primes; cp] 
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
                child_prime_old = vect_prime(child)
            
                nestedADMM!(child, qToChi; tol = tol, max_iter = max_iter, λ = λ)

                child_prime = vect_prime(child)
                primal_res = node.prime[child.ID] - child_prime
                dual_res = (child_prime - child_prime_old) / λ
                node.dual[child.ID] += primal_res
                push!(ter, max(norm(primal_res, Inf), norm(dual_res, Inf)))

                com_cost!(child, child_prime, 1) #Child sent its prime variable to node
            end

            if maximum(abs.(ter)) < tol
                if node.parent === nothing
                    println("nADMM converged after $iteration iterations in root")
                end
                
                break
            end
        end
    else
        update_leaf!(node, top_query; λ = λ)
    end
end
