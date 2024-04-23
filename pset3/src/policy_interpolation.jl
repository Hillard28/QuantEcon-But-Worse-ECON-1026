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
function u(c, gamma)
    return (c^(1 - gamma)) / (1 - gamma)
end

function u_c(c, gamma)
    return c^(-gamma)
end

function u_c_inv(c, gamma)
    return c^(-1/gamma)
end

function Eu_c(states, y, c, gamma, P)
    N = length(states)

    i_y = findfirst(state -> state == y, states)
    P_y = P[i_y, :]
    return P_y' * u_c.(c, gamma)
end

function cEuler(states, y, c, c1, R, beta, gamma, P)
    lhs = u_c(c, gamma)
    rhs = beta * R * Eu_c(states, y, c1, gamma, P)
    return lhs - rhs
end

function grid_distance(min, max, minval, maxval, i, shape=1)
    minval + (maxval - minval)*((i - min) / (max - min))^shape
end

function grid_locate(grid, grid_length, point; reverse=false)
    if grid[1] >= point
        return 1
    elseif grid[grid_length] <= point
        return grid_length
    else
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

function bisection(
    states,
    P,
    grid,
    grid_length,
    spoint,
    mpoint,
    epoint,
    a,
    y,
    A2,
    R,
    beta,
    gamma,
    nu;
    tolerance=1e-4,
    print_output=false
    )

    pos = grid_locate(grid, grid_length, spoint; reverse=false)
    if typeof(pos) == Tuple{Int64, Int64}
        A2m = A2[pos[1], :] .+ (spoint - grid[pos[1]]) .* ((A2[pos[2], :] .- A2[pos[1], :]) ./ (grid[pos[2]] - grid[pos[1]]))
    else
        A2m = A2[pos, :]
    end
    
    c = R*a + y - spoint
    c1 = R*spoint .+ (states .- A2m)
    if c >= 0.0 && minimum(c1 .>= 0.0)
        spointE = cEuler(states, y, c, c1, R, beta, gamma, P)
    else
        if print_output
            println("Negative consumption at startpoint: $spoint.")
        end
        return missing
    end
    if spointE > 0.0
        if print_output
            println("Converged to interior solution at startpoint: $spoint.")
        end
        return spoint
    else
        pos = grid_locate(grid, grid_length, mpoint; reverse=false)
        if typeof(pos) == Tuple{Int64, Int64}
            A2m = A2[pos[1], :] .+ (mpoint - grid[pos[1]]) .* ((A2[pos[2], :] .- A2[pos[1], :]) ./ (grid[pos[2]] - grid[pos[1]]))
        else
            A2m = A2[pos, :]
        end

        c = R*a + y - mpoint
        c1 = R*mpoint .+ (states .- A2m)
        if c >= 0.0 && minimum(c1 .>= 0)
            mpointE = cEuler(states, y, c, c1, R, beta, gamma, P)
        else
            if print_output
                println("Negative consumption at midpoint: $mpoint.")
            end
            return bisection(states, P, grid, grid_length, spoint, (spoint + mpoint)/2, mpoint, a, y, A2, R, beta, gamma, nu; tolerance=tolerance, print_output=print_output)
        end
        if 0.0 <= mpointE && mpointE <= 0.0 + tolerance
            if print_output
                println("Converged to interior solution at midpoint: $mpoint.")
            end
            return mpoint
        elseif mpointE > 0.0 + tolerance
            return bisection(states, P, grid, grid_length, spoint, (spoint + mpoint)/2, mpoint, a, y, A2, R, beta, gamma, nu; tolerance=tolerance, print_output=print_output)
        else
            pos = grid_locate(grid, grid_length, epoint; reverse=false)
            if typeof(pos) == Tuple{Int64, Int64}
                A2m = A2[pos[1], :] .+ (epoint - grid[pos[1]]) .* ((A2[pos[2], :] .- A2[pos[1], :]) ./ (grid[pos[2]] - grid[pos[1]]))
            else
                A2m = A2[pos, :]
            end

            c = R*a + y - epoint
            c1 = R*epoint .+ (states .- A2m)
            if c >= 0.0 && minimum(c1 .>= 0)
                epointE = cEuler(states, y, c, c1, R, beta, gamma, P)
            else
                if print_output
                    println("Negative consumption at endpoint: $epoint.")
                end
                return bisection(states, P, grid, grid_length, mpoint, (mpoint + epoint)/2, epoint, a, y, A2, R, beta, gamma, nu; tolerance=tolerance, print_output=print_output)
            end
            if epointE < 0.0
                if print_output
                    println("No solution found, consider expanding grid.")
                end
                return missing
            elseif 0.0 <= epointE && epointE <= 0.0 + tolerance
                if print_output
                    println("Converged to interior solution at endpoint: $epoint.")
                end
                return epoint
            else
                return bisection(states, P, grid, grid_length, mpoint, (mpoint + epoint)/2, epoint, a, y, A2, R, beta, gamma, nu; tolerance=tolerance, print_output=print_output)
            end
        end
    end
end

# Add check for valid consumption
function pfi_interpolation(
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
    tolerance=1e-4,
    print_output=false
    )

    N = length(states)

    A = Array{Union{Float64, Missing}}(undef, grid_length)
    A[1] = phi
    A[grid_length] = grid_max
    for i = 2:grid_length-1
        A[i] = grid_distance(1, grid_length, phi, grid_max, i, nu)
    end

    Ay1 = Array{Union{Float64, Missing}}(undef, (grid_length, N))
    Ay2 = Array{Union{Float64, Missing}}(undef, (grid_length, N))
    for j = 1:N
        Ay1[:, j] = A
        Ay2[:, j] = A
    end

    for iteration = 1:max_iterations
        if print_output
            println("Iteration: ", iteration)
        end
        for i = 1:grid_length
            for j = 1:N
                if print_output
                    println("i: $i, j: $j")
                end
                spoint = phi
                epoint = grid_max
                mpoint = (spoint + epoint) / 2
                iguess = A[i]
                value = bisection(states, P, A, grid_length, spoint, iguess, epoint, A[i], states[j], Ay2, R, beta, gamma, nu; tolerance=tolerance, print_output=print_output)
                if typeof(value) != Missing
                    Ay1[i, j] = value
                end
            end
        end
        error = maximum(abs.(skipmissing(Ay1 .- Ay2)))
        if error <= tolerance
            if print_output
                println("\nError (", round(error, digits=4), ") within tolerance in ", iteration, " iterations, exiting.\n")
            end
            break
        elseif iteration == max_iterations
            println("Interpolation failed to converge.\nErrors:")
            if print_output
                for i = 1:grid_length
                    println(round.(abs.(skipmissing(Ay1[i, :] .- Ay2[i, :])), digits=4))
                end
            end
        else
            if print_output
                println("\nError (", round(error, digits=4), ") outside of tolerance, updating Ay2.\nErrors:")
            end
            Ay2[:, :] = Ay1[:, :]
        end
    end

    return A, Ay1
end


#==========================================================================================
# Problem 2 (Interpolation)
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
ν = 1
a_M = 80

# Markov chain parameters
N = 5

markov = rouwenhorst(w̄, vₑ, 5, ρ)

states = exp.(markov.states)
Θ = markov.theta

A_policy, Ay1_policy = pfi_interpolation(
    states,
    Θ,
    M,
    a_M,
    ϕ,
    ν,
    R,
    β,
    γ;
    max_iterations=1000,
    tolerance=1e-8,
    print_output=false
)

for i = 1:M
    println(round(A_policy[i], digits=1), ": ", round.(Ay1_policy[i, :], digits=2))
end


#==========================================================================================
# Simulation
==========================================================================================#
periods = 100
Y = exp.(simulate!(markov, periods, 1000, 0; random_state=-1))
A = Array{Union{Float64, Missing}}(undef, periods)
#A[1] = A_policy[rand(1:M)]
A[1] = 0.0
A1 = Array{Union{Float64, Missing}}(undef, periods)
A1[1] = Ay1_policy[
    findfirst(asset -> asset == A[1], A_policy),
    findfirst(state -> state == Y[1], states)
]
for t = 2:periods
    A[t] = A1[t - 1]
    j = findfirst(state -> state == Y[t], states)
    for k = 1:length(A_policy)-1
        if A_policy[k] == A[t]
            A1[t] = Ay1_policy[k, j]
            break
        elseif A_policy[k+1] == A[t]
            A1[t] = Ay1_policy[k+1, j]
            break
        elseif A_policy[k] < A[t] && A[t] < A_policy[k+1]
            A1[t] = Ay1_policy[k, j] + (A[t] - A_policy[k]) *
                (Ay1_policy[k+1, j] - Ay1_policy[k, j]) /
                (A_policy[k+1] - A_policy[k])
            break
        end
    end
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
=#

sim_plot = plt.plot(Y, label="Endowment")
sim_plot = plt.plot!(A1, label="Savings")
sim_plot = plt.plot!(C, label="Consumption")
plt.display(sim_plot)
#=
Cshare_income = C ./ (R.*A .+ Y)
Cshare_endowment = C ./ Y

csim_plot = plt.plot(Cshare_income, label="Y + R*A")
csim_plot = plt.plot!(Cshare_endowment, label="Y")
plt.display(csim_plot)
=#
error = Array{Union{Float64, Missing}}(undef, periods)
for t = 1:periods
    j = findfirst(state -> state == Y[t], states)
    if Ay1_policy[1, j] == A1[t]
        A2 = Ay1_policy[1, :]
        c1 = R*A1[t] .+ states .- A2
        error[t] = abs(1 - u_c_inv(β * R * Eu_c(states, Y[t], c1, γ, Θ), γ) / C[t])
    else
        for i = 1:M - 1
            if Ay1_policy[i+1, j] == A1[t]
                A2 = Ay1_policy[i+1, :]
                c1 = R*A1[t] .+ states .- A2
                error[t] = abs(1 - u_c_inv(β * R * Eu_c(states, Y[t], c1, γ, Θ), γ) / C[t])
                break
            elseif Ay1_policy[i+1, j] > A1[t]
                A2 = Ay1_policy[i, :] .+ (A1[t] - Ay1_policy[i, j]) .* ((Ay1_policy[i+1, :] .- Ay1_policy[i, :]) ./ (Ay1_policy[i+1, j] - Ay1_policy[i, j]))
                c1 = R*A1[t] .+ states .- A2
                error[t] = abs(1 - u_c_inv(β * R * Eu_c(states, Y[t], c1, γ, Θ), γ) / C[t])
                break
            end
        end
    end
end
println("Mean interpolation error: ", stats.mean(error))
