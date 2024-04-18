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
    σ = sqrt(variance)
    v = variance
    μ = mean
    states = Array{Float64}(undef, N)
    states[N] = m*sqrt(v / (1 - rho^2))
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
                Θ[i, j] = dist.cdf(D, (states[j] + d/2 - rho*states[i])/σ)
            elseif j == N
                Θ[i, j] = 1 - dist.cdf(D, (states[j] - d/2 - rho*states[i])/σ)
            else
                Θ[i, j] = dist.cdf(D, (states[j] + d/2 - rho*states[i])/σ) -
                    dist.cdf(D, (states[j] - d/2 - rho*states[i])/σ)
            end
        end
    end
    
    if print_output
        println("Θ:")
        for i = 1:N
            println(round.(Θ[i, :], digits=3))
        end
    end
    
    states .+= μ / (1 - rho)

    return MarkovChain(Θ, states, missing, missing, missing)
end

function rouwenhorst(
    mean,
    variance,
    N,
    rho;
    print_output=false
    )
    v = variance
    μ = mean

    ψ = sqrt(v / (1 - rho^2)) * sqrt(N - 1)

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

    states .+= μ / (1 - rho)
    
    return MarkovChain(Θ, states, missing, missing, missing)
end

#==========================================================================================
# Utility and asset grid functions
==========================================================================================#
function u(c, gamma)
    return c^(1 - gamma) / (1 - gamma)
end

function u_c(c, gamma)
    return c^(-gamma)
end

function Eu_c_y(states, y, a1, A2_y, R, gamma, P)
    N = length(states)

    i_y = findfirst(state -> state == y, states)
    P_y = P[i_y, :]
    global u_c_y = 0.0
    for j = 1:N
        u_c_y += P_y[j] * u_c(R*a1 - A2_y[j] + states[j], gamma)
    end
    return u_c_y
end

function cEuler(states, y, a, a1, A2_y, R, beta, gamma, P)
    return u_c(R*a + y - a1, gamma) - beta * R * Eu_c_y(states, y, a1, A2_y, R, gamma, P)
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

    A1 = Array{Union{Float64, Missing}}(undef, grid_length)
    A1[1] = phi
    A1[grid_length] = grid_max
    for i = 2:grid_length-1
        A1[i] = A1[1] +
            (A1[grid_length] - A1[1])*((i - 1) / (grid_length - 1))^nu
    end

    Ay1 = Array{Union{Float64, Missing}}(undef, (grid_length, N))
    for i = 1:grid_length
        for j = 1:N
            Ay1[i, j] = missing
        end
    end

    Ay2 = Array{Union{Float64, Missing}}(undef, (grid_length, N))
    for i = 1:grid_length
        for j = 1:N
            Ay2[i, j] = A1[i]
        end
    end

    for iteration = 1:max_iterations
        for i = 1:grid_length
            for j = 1:N
                ijk = cEuler(
                    states,
                    states[j],
                    A[i],
                    A1[1],
                    Ay2[1, :],
                    R,
                    beta,
                    gamma,
                    P
                )
                if print_output
                    println(i, ", ", j, ": ", ijk)
                end
                if ijk >= 0
                    Ay1[i, j] = A1[1]
                else
                    for k = 2:grid_length
                        if print_output
                            println("After (", k, "): ", ijk)
                        end
                        ijk1 = cEuler(
                            states,
                            states[j],
                            A[i],
                            A1[k],
                            Ay2[k, :],
                            R,
                            beta,
                            gamma,
                            P
                        )
                        if ijk1 >= 0
                            if abs(ijk1) <= abs(ijk)
                                Ay1[i, j] = A1[k]
                            else
                                Ay1[i, j] = A1[k-1]
                            end
                            break
                        else
                            ijk = ijk1
                            if print_output
                                println("Before (", k, "): ", ijk)
                            end
                        end
                    end
                end
            end
        end
        if maximum(skipmissing(Ay1 .== Ay2))
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
γ = 2.0
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
a_M = 20

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
    max_iterations=1000,
    print_output=false
)

for i = 1:M
    println(round.(Ay1_policy[i, :], digits=3))
end

#==========================================================================================
# Simulation
==========================================================================================#
periods = 100
Y = exp.(simulate!(markov, periods, 1000, 0; random_state=42))
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

Cshare_income = C ./ (R.*A .+ Y)
Cshare_endowment = C ./ Y

csim_plot = plt.plot(Cshare_income, label="Y + R*A")
csim_plot = plt.plot!(Cshare_endowment, label="Y")
plt.display(csim_plot)
