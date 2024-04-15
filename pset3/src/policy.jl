#==============================================================================
# ECON-GA 1026 Problem Set 3
# Ryan Gilland
==============================================================================#
import LinearAlgebra as linalg
import Statistics as stats
import Distributions as dist
import QuantEcon as qe
import Random as random
import Plots as plt

#==============================================================================
# Utility functions
==============================================================================#
function plot_series(y; ylim, display=true)
    x = range(0, length(y)-1, length=length(y))
    series_plot = plt.plot(x, y; ylims=(-ylim, ylim))
    if display
        plt.display(series_plot)
    end
end

function plot_histogram(y; bins=:auto, display=true)
    histogram_plot = plt.histogram(y, bins=bins)
    if display
        plt.display(histogram_plot)
    end
end

function print_theta(theta, N)
    for i = 1:N
        println(round.(theta[(i-1)*N+1:i*N], digits=3))
    end
end

#==============================================================================
# Markov Chain
==============================================================================#
mutable struct MarkovChain
    theta::Array{Float64}
    states::Array{Float64}
    mean::Union{Float64, Missing}
    var::Union{Float64, Missing}
    std::Union{Float64, Missing}
end

function simulate!(markovchain, periods=1000, burn=0, replications=0; random_state)
    if random_state >= 0
        random.seed!(random_state)
    end
    N = length(markovchain.states)
    
    # Configure CDF
    Θ = Array{dist.Categorical{Float64, Vector{Float64}}}(undef, N)
    for i = 1:N
        Θ[i] = dist.Categorical(markovchain.theta[(i-1)*N+1:i*N])
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

#==============================================================================
# Tauchen / Rouwenhorst
==============================================================================#
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
    Θ = Array{Float64}(undef, N^2)
    for i = 1:N
        for j = 1:N
            if j == 1
                Θ[(i-1)*N + j] = dist.cdf(D, (states[j] + d/2 - rho*states[i])/σ)
            elseif j == N
                Θ[(i-1)*N + j] = 1 - dist.cdf(D, (states[j] - d/2 - rho*states[i])/σ)
            else
                Θ[(i-1)*N + j] = dist.cdf(D, (states[j] + d/2 - rho*states[i])/σ) -
                    dist.cdf(D, (states[j] - d/2 - rho*states[i])/σ)
            end
        end
    end
    
    if print_output
        println("Θ:")
        for i = 1:N
            println(round.(Θ[(i-1)*N+1:i*N], digits=3))
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
    Θₙ = Array{Float64}(undef, 4)
    Θₙ[1] = p
    Θₙ[2] = 1 - p
    Θₙ[3] = 1 - q
    Θₙ[4] = q
    Θ = Θₙ
    if N > 2
        global Θ
        global Θₙ
        for n = 3:N
            global Θ
            global Θₙ
            Θ = zeros(n^2)
            for i = 1:n
                for j = 1:n
                    if i <= n-1 && j <= n-1
                        Θ[(i-1)*n + j] += p*Θₙ[(i-1)*(n-1) + j]
                    end
                    if i <= n-1 && j >= 2
                        Θ[(i-1)*n + j] += (1 - p)*Θₙ[(i-1)*(n-1) + (j-1)]
                    end
                    if i >= 2 && j <= n-1
                        Θ[(i-1)*n + j] += (1 - q)*Θₙ[(i-2)*(n-1) + j]
                    end
                    if i >= 2 && j >= 2
                        Θ[(i-1)*n + j] += q*Θₙ[(i-2)*(n-1) + (j-1)]
                    end
                end
            end
            for i = 2:n-1
                for j = 1:n
                    Θ[(i-1)*n + j] /= 2
                end
            end
            Θₙ = Θ
        end
    end
    
    if print_output
        println("Θ:")
        for i = 1:N
            println(round.(Θ[(i-1)*N+1:i*N], digits=3))
        end
    end

    states .+= μ / (1 - rho)
    
    return MarkovChain(Θ, states, missing, missing, missing)
end

#==============================================================================
# Problem 2
==============================================================================#
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
Dₑ = dist.Normal(μₑ, σₑ)
ρ = 0.90
w̄ = -vₑ / (2*(1 + ρ))
v = σₑ / (1 - ρ^2)
σ = sqrt(v)
μ = w̄ / (1 - ρ)

# Grid parameters
M = 500
ν = 1

# Markov chain parameters
N = 5

# Value and utility functions
function u(c, gamma)
    return c^(1 - gamma) / (1 - gamma)
end

function u_c(c, gamma)
    return c^(-gamma)
end

function Eu_c_y(states, y, c, a1, a2, R, gamma, P)
    N = length(states)
    i_y = findfirst(state -> state == y, states)
    P_y = P[(i_y-1)*N+1:i_y*N]
    u_c_y = Array{Float64}(undef, N)
    for i = 1:N
        u_c_y[i] = u_c(R*a1 - a2 + states[i], gamma)
    end
    return P_y' * u_c_y
end

function cEuler(states, y, a, c, a1, a2, R, beta, gamma, P)
    return u_c(R*a + y - a1, gamma) -
        beta * R * Eu_c_y(states, y, c, a1, a2, R, gamma, P)
end

markov = rouwenhorst(w̄, vₑ, 5, ρ; print_output=false)
states = markov.states
P = markov.theta

A = Array{Float64}(undef, M)
A[1] = ϕ
A[M] = 1
for i = 2:M-1
    A[i] = A[1] + (A[M] - A[1])*((i - 1) / (M - 1))^ν
end


