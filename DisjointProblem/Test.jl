using LinearAlgebra
using Zygote
using ProximalOperators
using ProximalAlgorithms
using DifferentiationInterface: AutoZygote

# Setup
n = 10
γ = randn(n)
λ = 0.1

# Smooth function f(x) = 0.5 * ||x - γ||^2
f = ProximalAlgorithms.AutoDifferentiable(
    x -> 0.5 * dot(x - γ, x - γ),
    AutoZygote()
)

# Proximal operator (non-smooth part): g(x) = λ * ||x||₁
g = NormL1(λ)  # 

# Initial point
x0 = zeros(n)

# Set up and run PANOC
panoc = ProximalAlgorithms.PANOC(maxit = 500, tol = 1e-6, verbose = true)

# This is now fully correct
solution, iterations = panoc(f = f, g = g, x0 = x0)

# Display result
println("\n✅ Optimal solution x:\n", solution)
println("🔁 Total iterations: ", iterations)
