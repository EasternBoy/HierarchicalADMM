using LinearAlgebra

"""
    admm_solver(f_grad, g_prox, A, B, c; rho=1.0, max_iter=1000, tol=1e-4)

Solve the convex optimization problem:
    min  f(x) + g(z)
    s.t. Ax + Bz = c
using the Alternating Direction Method of Multipliers (ADMM).

- `f_grad(x)`: Returns gradient ∇f(x) of the smooth function f(x).
- `g_prox(v, λ)`: Proximal operator for the function g(z) with step size λ.
- `A, B, c`: Constraint matrices.
- `rho`: Augmented Lagrangian penalty parameter.
"""
function admm_solver(f_grad, g_prox, A, B, c; rho=1.0, max_iter=1000, tol=1e-4)
    m, n = size(A)
    _, p = size(B)

    # Initialize variables
    x = zeros(n)
    z = zeros(p)
    u = zeros(m)

    # Precompute matrix decomposition for x-update
    M = A' * A + rho * I(n)
    L = cholesky(M)

    for k in 1:max_iter
        # x-update: Solve (A'A + ρI)x = A'(c - Bz - u) - ∇f(x)
        x = L \ (A' * (c - B * z - u) - f_grad(x))

        # z-update: Solve z = prox_g((B'x + u - c) / ρ, 1/ρ)
        z_old = z
        z = g_prox((B' * x + u - c) / rho, 1 / rho)

        # Dual update: u = u + ρ(Ax + Bz - c)
        u += rho * (A * x + B * z - c)

        # Convergence check
        r_norm = norm(A * x + B * z - c, 2)
        s_norm = norm(rho * (z - z_old), 2)

        if r_norm < tol && s_norm < tol
            println("ADMM converged in $k iterations.")
            break
        end
    end

    return x, z
end

# Example: Lasso Regression (Convex Optimization)
function test_admm_lasso()
    # Define problem: LASSO (min ||Ax - b||² + λ||x||₁)
    A = [3.0 1.0; 1.0 2.0]
    b = [1.0, 2.0]
    lambda = 0.1

    function f_grad(x)
        return 2 * A' * (A * x - b)  # Gradient of ||Ax - b||²
    end

    function g_prox(v, λ)
        return sign.(v) .* max.(abs.(v) .- λ, 0)  # Soft thresholding (L1 norm proximal)
    end

    B = I(2)  # Identity, since we want z = x
    c = zeros(2)

    x_opt, _ = admm_solver(f_grad, g_prox, A, B, c)
    println("Optimal solution: ", x_opt)
end

test_admm_lasso()