include("TreeGeneration.jl")

function prox!(node::linknode; λ = 0.2)  

    opti = JuMP.Model(Ipopt.Optimizer)
    set_silent(opti)

    x0 = ones(node.nV)
    @variable(opti, x[i = 1:node.nV], base_name = string("x",node.ID), start = x0[i])
    @constraint(opti, x .>= 1e-5)

    func = node.cost_func.val
    para = node.cost_func.para
    J = 0
    q = Float64[]

    
    if node.children === nothing #Leaf nodes
        q = node.parent.prime[node.ID] + node.parent.dual[node.ID] #query
        
        com_cost!(node.parent,q, 1)

        @variable(opti, sl[i=1:node.nV] .>= 0, base_name = string("sl", node.ID))
        J += sum(sl) + 1/(2λ)*(x - q)'*(x - q)
        @constraint(opti, sl .>= 1 ./x .- 1/para[1])

    elseif node.parent === nothing #Root node
        nc = length(node.children)
        
        for i in 1:nc
            q  = [q; vect_prime(node.children[i]) - node.dual[node.children[i].ID]]
        end
        # Received prime from all children

        J += func(x; para = para) + 1/(2λ)*dot(x - q, x - q)
    else #Middle node
        nc = length(node.children)

        par  = node.parent.prime[node.ID] + node.parent.dual[node.ID]

        com_cost!(node.parent, par, 1) #Received prime + dual from parent

        ch_pr = Float64[]
        for i in 1:nc
            ch_pr = [ch_pr;  vect_prime(node.children[i])]
        end
        # Received prime from all children

        q = [q; par - vect_dual(node) + ch_pr]

        η  = para[1]
        τ  = para[2]
        nc = para[3]
        horizon = Int64(node.nV/nc)
        @variable(opti, t[i=1:horizon] .>= 0, base_name = string("t",node.ID))
        for i in 1:horizon
            @constraint(opti, t[i] >= sum([x[(j-1)*horizon + i] for j in 1:nc]) - τ)
        end
        J += η*sum(t) + (1/λ)*dot(x - q/2, x - q/2)
    end

    @objective(opti, Min, J)
    JuMP.optimize!(opti)
    x = JuMP.value.(x)

    #Update prime variable
    if node.children === nothing #leaf node has only one
        # sl_opt = JuMP.value.(sl)
        node.prime[node.ID] = x
    else
        index = 1
        for i in 1:nc
            ID  = node.children[i].ID
            nVi = node.children[i].nV
            node.prime[ID] = x[index:(index+nVi-1)]
            index += nVi
        end
    end
end


function hADMM_JuMP!(node::linknode, ter::Vector{Float64}; λ = 0.2)
    node.iteration += 1
    #Update prime
    prox!(node; λ = λ)

    if node.children !== nothing
        for child in node.children
            child_prime_old = vect_prime(child)
            hADMM_JuMP!(child, ter; λ = λ)

            #Update dual
            prime_child = vect_prime(child)
            prime_res = node.prime[child.ID] - prime_child
            dual_res  = (prime_child - child_prime_old)/λ
            node.dual[child.ID] += prime_res

            # Receive a prime variable of one child for updating dual variable
            com_cost!(child, prime_child, 1)

            #Take the maximum residual error for stopping criteria
            push!(ter, max(norm(prime_res, Inf), norm(dual_res, Inf)))
        end
    end
end

function hADMM_JuMP(root::linknode; max_iter = 200, λ = 0.2)

    J_hADMM  = Float64[]
    traj_res = Float64[]

    push!(J_hADMM, total_cost(vect_prime(root), root))

    for step in 1:max_iter
        ter = Float64[]

        hADMM_JuMP!(root, ter; λ = λ)

        push!(J_hADMM, total_cost(vect_prime(root), root))

        if maximum(ter) < tol
            println("Terminates at step $step")
            return traj_res, J_hADMM
        end

        push!(traj_res, maximum(ter))
        if step % 10 == 0 println("Step $step: Primal-Dual stopping criteria = ", maximum(ter)) end
    end

    return traj_res, J_hADMM
end
