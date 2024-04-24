#==========================================================================================
# ECON-GA 1026 Problem Set 3
# Ryan Gilland
==========================================================================================#
import LinearAlgebra as linalg
import Statistics as stats
import Distributions as dist
import QuantEcon as qe
import Random as random
import Plots as plt

#==========================================================================================
# Markov Chain
==========================================================================================#
mutable struct MarkovChain
    theta::Array{Float64}
    states::Array{Float64}
    mean::Union{Float64, Missing}
    var::Union{Float64, Missing}
    std::Union{Float64, Missing}
end

function simulate!(markovchain, periods=1000, burn=0, replications=0; random_state=-1)
    if random_state >= 0
        random.seed!(random_state)
    end
    N = length(markovchain.states)
    
    # Configure CDF
    Θ = Array{dist.Categorical{Float64, Vector{Float64}}}(undef, N)
    for i = 1:N
        Θ[i] = dist.Categorical(markovchain.theta[i, :])
    end

    # Initialize index of realized states and run burn-in
    index_series = Array{Int64}(undef, periods)
    index_series[1] = rand(1:N)
    if burn > 0
        for _ = 1:burn
            index_series[1] = dist.rand(Θ[index_series[1]], 1)[1]
        end
    end

    # Run simulation
    for t = 2:periods
        index_series[t] = dist.rand(Θ[index_series[t-1]], 1)[1]
    end
    
    state_series = Array{Float64}(undef, periods)
    for t = 1:periods
        state_series[t] = markovchain.states[index_series[t]]
    end

    markovchain.mean = stats.mean(state_series)
    markovchain.var = stats.var(state_series)
    markovchain.std = stats.std(state_series)

    # If replication is specified, rerun simulation and take average of moments
    if replications > 0
        for replication = 1:replications
            index_series[1] = rand(1:N)
            if burn > 0
                for _ = 1:burn
                    index_series[1] = dist.rand(Θ[index_series[1]], 1)[1]
                end
            end

            for t = 2:periods
                index_series[t] = dist.rand(Θ[index_series[t-1]], 1)[1]
            end
            
            for t = 1:periods
                state_series[t] = markovchain.states[index_series[t]]
            end

            markovchain.mean += stats.mean(state_series)
            markovchain.var += stats.var(state_series)
            markovchain.std += stats.std(state_series)
        end
        markovchain.mean /= (replications + 1)
        markovchain.var /= (replications + 1)
        markovchain.std /= (replications + 1)
    end

    return state_series
end

#==========================================================================================
# Tauchen / Rouwenhorst
==========================================================================================#
function tauchen(
    mean,
    variance,
    N,
    m,
    rho;
    print_output=false
    )
    
    std = sqrt(variance)
    states = Array{Float64}(undef, N)
    states[N] = m*sqrt(variance / (1 - rho^2))
    states[1] = -states[N]
    if print_output
        println("States:")
        println(states[1])
    end
    d = (states[N] - states[1])/(N - 1)
    for i = 2:N-1
        states[i] = states[1] + (i - 1)*d
        if print_output
            println(states[i])
        end
    end
    if print_output
        println(states[N])
    end
    
    # Standard normal distribution for the normalized states
    D = dist.Normal(0, 1)
    
    # Transition matrix
    Θ = Array{Float64}(undef, (N, N))
    for i = 1:N
        for j = 1:N
            if j == 1
                Θ[i, j] = dist.cdf(D, (states[j] + d/2 - rho*states[i])/std)
            elseif j == N
                Θ[i, j] = 1 - dist.cdf(D, (states[j] - d/2 - rho*states[i])/std)
            else
                Θ[i, j] = dist.cdf(D, (states[j] + d/2 - rho*states[i])/std) -
                    dist.cdf(D, (states[j] - d/2 - rho*states[i])/std)
            end
        end
    end
    
    if print_output
        println("Θ:")
        for i = 1:N
            println(round.(Θ[i, :], digits=3))
        end
    end
    
    states .+= mean / (1 - rho)

    return MarkovChain(Θ, states, missing, missing, missing)
end

function rouwenhorst(
    mean,
    variance,
    N,
    rho;
    print_output=false
    )

    ψ = sqrt(variance / (1 - rho^2)) * sqrt(N - 1)

    states = Array{Float64}(undef, N)
    states[N] = ψ
    states[1] = -ψ
    if print_output
        println("States:")
        println(states[1])
    end
    d = (states[N] - states[1])/(N - 1)
    for i = 2:N-1
        states[i] = states[1] + (i - 1)*d
        if print_output
            println(states[i])
        end
    end
    if print_output
        println(states[N])
    end

    p = (1 + rho) / 2
    q = p
    
    # Transition matrix
    Θₙ = Array{Float64}(undef, (2, 2))
    Θₙ[1, 1] = p
    Θₙ[1, 2] = 1 - p
    Θₙ[2, 1] = 1 - q
    Θₙ[2, 2] = q
    Θ = Θₙ
    if N > 2
        global Θ
        global Θₙ
        for n = 3:N
            global Θ
            global Θₙ
            Θ = zeros((n, n))
            Θ[1:n-1, 1:n-1] += p .* Θₙ
            Θ[1:n-1, 2:n] += (1 - p) .* Θₙ
            Θ[2:n, 1:n-1] += (1 - q) .* Θₙ
            Θ[2:n, 2:n] += q .* Θₙ
            Θ[2:n-1, :] ./= 2
            Θₙ = Θ
        end
    end
    
    if print_output
        println("Θ:")
        for i = 1:N
            println(round.(Θ[i, :], digits=3))
        end
    end

    states .+= mean / (1 - rho)
    
    return MarkovChain(Θ, states, missing, missing, missing)
end

#==========================================================================================
# Utility and asset grid functions
==========================================================================================#
# Computes u(c)
function u(c, gamma)
    if gamma == 1
        return log(c)
    else
        return (c^(1 - gamma)) / (1 - gamma)
    end
end

# Computes u_c(c)
function u_c(c, gamma)
    return c^(-gamma)
end

# Computes u_c(c)^(--1)
function u_c_inv(c, gamma)
    return c^(-1/gamma)
end

# Computes the expected value of u_c(c_1)
function Eu_c(states, y, c, gamma, P)
    N = length(states)

    i_y = findfirst(state -> state == y, states)
    P_y = P[i_y, :]
    return P_y' * u_c.(c, gamma)
end

# Computes the Euler equation
function cEuler(states, y, c, c1, R, beta, gamma, P)
    lhs = u_c(c, gamma)
    rhs = beta * R * Eu_c(states, y, c1, gamma, P)
    return lhs - rhs
end

# Computes the distance between grid points using a shape parameter
function grid_distance(min, max, minval, maxval, i, shape=1)
    minval + (maxval - minval)*((i - min) / (max - min))^shape
end

# Locates grid points surrounding a given value
function grid_locate(grid, grid_length, point; reverse=false)
    if grid[1] >= point
        return 1
    elseif grid[grid_length] <= point
        return grid_length
    else
        # Make search quicker if near end of grid
        if reverse
            for i = grid_length:-1:2
                if grid[i-1] == point
                    return i-1
                elseif grid[i-1] < point
                    return (i-1, i)
                end
            end
        else
            for i = 1:grid_length - 1
                if grid[i+1] == point
                    return i+1
                elseif grid[i+1] > point
                    return (i, i+1)
                end
            end
        end
    end
end

function interpolate_vec(grid, grid_int, point, pos)
    return grid_int[pos[1], :] .+ (point - grid[pos[1]]) .* ((grid_int[pos[2], :] .- grid_int[pos[1], :]) ./ (grid[pos[2]] - grid[pos[1]]))
end

function interpolate(grid, grid_int, point, pos)
    return grid_int[pos[1]] .+ (point - grid[pos[1]]) .* ((grid_int[pos[2]] .- grid_int[pos[1]]) ./ (grid[pos[2]] - grid[pos[1]]))
end

function pfi_discretization(
    states,
    P,
    grid_length,
    grid_max,
    phi,
    nu,
    R,
    beta,
    gamma;
    max_iterations=1000,
    print_output=false
    )

    N = length(states)

    A = Array{Union{Float64, Missing}}(undef, grid_length)
    A[1] = phi
    A[grid_length] = grid_max
    for i = 2:grid_length-1
        A[i] = A[1] +
            (A[grid_length] - A[1])*((i - 1) / (grid_length - 1))^nu
    end

    Ay1 = Array{Union{Float64, Missing}}(undef, (grid_length, N))
    Ay2 = Array{Union{Float64, Missing}}(undef, (grid_length, N))
    for j = 1:N
        Ay1[:, j] = A
    end
    Ay2[:, :] = Ay1[:, :]

    for iteration = 1:max_iterations
        for i = 1:grid_length
            for j = 1:N
                c = R*A[i] + states[j] - A[1]
                c1 = R*A[1] .+ states .- Ay2[1, :]
                ijk = cEuler(
                    states,
                    states[j],
                    c,
                    c1,
                    R,
                    beta,
                    gamma,
                    P
                )
                if ijk >= 0
                    Ay1[i, j] = A[1]
                else
                    for k = 2:grid_length
                        c = R*A[i] + states[j] - A[k]
                        c1 = R*A[k] .+ states .- Ay2[k, :]
                        ijk1 = cEuler(
                            states,
                            states[j],
                            c,
                            c1,
                            R,
                            beta,
                            gamma,
                            P
                        )
                        if ijk1 >= 0
                            if abs(ijk1) <= abs(ijk)
                                Ay1[i, j] = A[k]
                            else
                                Ay1[i, j] = A[k-1]
                            end
                            break
                        else
                            ijk = ijk1
                        end
                    end
                end
            end
        end
        if minimum(skipmissing(Ay1 .== Ay2))
            if print_output
                println("Converged in $iteration iterations.")
            end
            break
        else
            Ay2[:, :] = Ay1[:, :]
        end
        if iteration == max_iterations
            if print_output
                println("Discretization failed to converge.")
            end
        end
    end

    return A, Ay1
end

#==========================================================================================
# Problem 2 (Discretization)
==========================================================================================#
# Consumption-savings parameters
ϕ = 0.0
γ = 2
β = 0.95
r = 0.02
R = 1 + r

# Income process parameters
vₑ = 0.06
σₑ = sqrt(vₑ)
μₑ = 0
ρ = 0.90
w̄ = -vₑ / (2*(1 + ρ))
v = σₑ / (1 - ρ^2)
σ = sqrt(v)
μ = w̄ / (1 - ρ)

# Grid parameters
M = 100
ν = 3
a_M = 50

# Markov chain parameters
N = 5

markov = rouwenhorst(w̄, vₑ, 5, ρ)

states = exp.(markov.states)
P = markov.theta

A_policy, Ay1_policy = pfi_discretization(
    states,
    P,
    M,
    a_M,
    ϕ,
    ν,
    R,
    β,
    γ;
    max_iterations=100,
    print_output=true
)

for i = 1:M
    println(round.(Ay1_policy[i, :], digits=3))
end

#==========================================================================================
# Simulation
==========================================================================================#
periods = 100000
Y = exp.(simulate!(markov, periods, 1000, 0; random_state=28))
A = Array{Union{Float64, Missing}}(undef, periods)
#A[1] = A_policy[rand(1:M)]
A[1] = 0.0
A1 = Array{Union{Float64, Missing}}(undef, periods)
A1[1] = Ay1_policy[findfirst(asset -> asset == A[1], A_policy), findfirst(state -> state == Y[1], states)]
for t = 2:periods
    A[t] = A1[t - 1]
    j = findfirst(state -> state == Y[t], states)
    i = findfirst(asset -> asset == A[t], A_policy)
    A1[t] = Ay1_policy[i, j]
end

C = Array{Union{Float64, Missing}}(undef, periods)
for t = 1:periods
    C[t] = Y[t] + R*A[t] - A1[t]
end
#=
for t = 1:periods
    println(
        "Y[", t, "]: ", round(Y[t], digits=3),
        "\tA[", t, "]: ", round(A[t], digits=3),
        "\tA'[", t, "]: ", round(A1[t], digits=3),
        "\tC[", t, "]: ", round(C[t], digits=3),
    )
end

sim_plot = plt.plot(Y, label="Endowment")
sim_plot = plt.plot!(A1, label="Savings")
sim_plot = plt.plot!(C, label="Consumption")
plt.display(sim_plot)

sim_plot = plt.plot(Y, label="Endowment")
sim_plot = plt.plot!(C, label="Consumption")
plt.display(sim_plot)

Cshare_income = C ./ (R.*A .+ Y)
Cshare_endowment = C ./ Y

csim_plot = plt.plot(Cshare_income, label="Y + R*A")
csim_plot = plt.plot!(Cshare_endowment, label="Y")
plt.display(csim_plot)
=#
error = Array{Union{Float64, Missing}}(undef, periods)
for t = 1:periods
    i = findfirst(asset -> asset == A1[t], A_policy)
    A2 = Ay1_policy[i, :]
    c1 = R*A1[t] .+ states .- A2
    error[t] = abs(1 - u_c_inv(β * R * Eu_c(states, Y[t], c1, γ, Θ), γ) / C[t])
end
println("Mean discretization error: ", stats.mean(error))
