using BenchmarkTools
using CairoMakie
using DynamicPolynomials
using Hypatia
using JuMP
using JSServe
using LinearAlgebra
using Makie.GeometryBasics
using MosekTools
using Pajarito
using SumOfSquares
using WGLMakie

struct Box
  xl::Float64
  xu::Float64
  yl::Float64
  yu::Float64
end

# set box constraints
boxes = Array{Box}(UndefInitializer(), 3)
box₁ = Box(-10.00, 0.00, -10.00, 20.00)
box₂ = Box(-5.00, 20.00, 7.50, 30.00)
box₃ = Box(15.00, 25.00, -5.00, 20.00)
boxes[1] = box₁
boxes[2] = box₂
boxes[3] = box₃

function cost(x)
    return x^2
end

function center(box::Box)
    return [(box.xu - box.xl)/2; (box.yu - box.yl)/2]
end
    
# set obstacles
obstacles = Array{Box}(UndefInitializer(), 4)
obstacle₁ = Box(-10.00, -5.00, 20.00, 30.00)
obstacle₂ = Box(20.00, 25.00, 20.00, 30.00)
obstacle₃ = Box(0.00, 15.00, -10.00, 7.50)
obstacle₄ = Box(15.00, 25.00, -10.00, -5.00)
obstacles[1] = obstacle₁
obstacles[2] = obstacle₂
obstacles[3] = obstacle₃
obstacles[4] = obstacle₄

# bottom, left, top, right to define intersection of half-spaces
a = [0 -1; -1 0; 0 1; 1 0]
# analytic centers for each region
x̂₁ = center(boxes[1])
x̂₂ = center(boxes[2])
x̂₃ = center(boxes[3])
# offsets of hyperplanes from origin for each box
b₁ = [-10; -10; 20; 0]
b₂ = [7.5; -5; 30; 20]
b₃ = [-5; 15.0; 20.0; 25]

function polygon(box::Box)
    return Polygon(Point2f[(box.xl, box.yl), (box.xl, box.yu), (box.xu, box.yu), (box.xu, box.yl)])
end

k = 20
X₀ = Dict()
X₀[:x], X₀[:y] = -3.5, -3.0
X₀′ = Dict()
X₀′[:x], X₀′[:y] = 1e0, 1e0
X₀″ = Dict()
X₀″[:x], X₀″[:y] = 1, 1
X₁ = Dict()
## X₁[:x], X₁[:y] = 15, 15
X₁[:x], X₁[:y] = 15, 20
X₁′ = Dict()
X₁′[:x], X₁′[:y] = 1e0, 1e0
X₁″ = Dict()
X₁″[:x], X₁″[:y] = 1, 1

function d(x)
    return mapreduce(x->sum((b₁-a*(x̂₁-x)).^2 + (b₂-a*(x̂₂-x)).^2 + (b₃-a*(x̂₃-x)).^2), +, eachcol(x'))
end

function solve_for_optimal_path(λ, N)
  model = SOSModel(optimizer_with_attributes(Pajarito.Optimizer,
  "conic_solver" => optimizer_with_attributes(Hypatia.Optimizer),
  "oa_solver" => optimizer_with_attributes(Mosek.Optimizer)))
  set_attribute(model, "iteration_limit", 38400)

  r = 3 # order of polynomial
  @polyvar(t)
  Z = monomials([t], 0:r)
  @variable(model, H[1:N, boxes], Bin)

  p = Dict()
  # Reformulate disjunctive constraint using big M method,
  # see https://optimization.cbe.cornell.edu/index.php?title=Disjunctive_inequalities#Big-M_Method
  # or https://en.wikipedia.org/wiki/Big_M_method
  Mxl = -10.0
  Mxu = 25.0
  Myl = -10.0
  Myu = 30.0

  # setup the polynomials so they satisfy the box constraints and each polynomial can be a member of only a single segment
  T = collect(0:1:N)
  for j in 1:N
    @constraint(model, sum(H[j, box] for box in boxes) == 1)
    p[(:x, j)] = @variable(model, [1:1], SumOfSquares.Poly(Z))
    p[(:y, j)] = @variable(model, [1:1], SumOfSquares.Poly(Z))
    S = @set t >= T[j] && t <= T[j+1]
    for box in boxes
      xl, xu, yl, yu = box.xl, box.xu, box.yl, box.yu
      @constraint(model, p[(:x, j)][1] >= Mxl + (xl-Mxl)*H[j, box], domain = S)
      @constraint(model, p[(:x, j)][1] <= Mxu + (xu-Mxu)*H[j, box], domain = S)
      @constraint(model, p[(:y, j)][1] >= Myl + (yl-Myl)*H[j, box], domain = S)
      @constraint(model, p[(:y, j)][1] <= Myu + (yu-Myu)*H[j, box], domain = S)
    end
  end
    print(p[(:x, 1)][1])

  # formulate optimization problem
  for ax in (:x, :y)
    @constraint(model, p[(ax, 1)][1]([0]) == X₀[ax])
    @constraint(model, differentiate(p[ax, 1][1], t, 1)([0]) == X₀′[ax])
    @constraint(model, differentiate(p[ax, 1][1], t, 2)([0]) == X₀″[ax])
    for j in 1:N-1
      @constraint(model, p[(ax, j)][1]([T[j+1]]) == p[(ax, j+1)][1]([T[j+1]]))
      @constraint(model, differentiate(p[(ax, j)][1], t, 1)([T[j+1]]) == differentiate(p[(ax, j+1)][1], t, 1)([T[j+1]]))
      # @constraint(model, differentiate(p[(ax, j)][1], t, 2)([T[j+1]]) == differentiate(p[(ax, j+1)][1], t, 2)([T[j+1]]))
    end
    @constraint(model, p[(ax, N)][1]([T[N]]) == X₁[ax])
    @constraint(model, differentiate(p[ax, N][1], t, 1)([T[N]]) == X₁′[ax])
    @constraint(model, differentiate(p[ax, N][1], t, 2)([T[N]]) == X₁″[ax])
  end

  # need to get variables in a form compatible for calling function d.
  x = []
  y = []
  for n in 2:N
    for i in T[n-1]:(T[n]-T[n-1])/k:T[n]
      xₜ, yₜ = (p[(:x, n-1)])[1](i), (p[(:y, n-1)])[1](i)
      push!(x, xₜ)
      push!(y, yₜ)
    end
  end
  x = hcat(x, y)

  @variable(model, γ[keys(p)] >= 0)
  for (key, val) in p
    @constraint(model, γ[key] >= differentiate(val[1], t, 3))
    @constraint(model, γ[key] >= -differentiate(val[1], t, 3))
  end

  @objective(model, Min, sum(γ) + λ*d(x))
  # solve the problem
  solve_time = @timed optimize!(model)
  if termination_status(model) != OPTIMAL
    return model, nothing, [], [], [], [], [], [], [], []
  end

  # show values
  x = Array{Float32}([])
  y = Array{Float32}([])
  x′ = Array{Float32}([])
  y′ = Array{Float32}([])
  x″ = Array{Float32}([])
  y″ = Array{Float32}([])
  x‴ = Array{Float32}([])
  y‴ = Array{Float32}([])
  for n in 2:N
    for i in T[n-1]:(T[n]-T[n-1])/20:T[n]
            if i == T[n]
                break
            end
      xₜ, yₜ = value.(p[(:x, n-1)])[1](i), value.(p[(:y, n-1)])[1](i)
      push!(x, xₜ)
      push!(y, yₜ)
      xₜ′, yₜ′ = differentiate(value.(p[(:x, n-1)])[1], t, 1)([i]), differentiate(value.(p[(:y, n-1)])[1], t, 1)([i])
      push!(x′, xₜ′)
      push!(y′, yₜ′)
      xₜ″, yₜ″ = differentiate(value.(p[(:x, n-1)])[1], t, 2)([i]), differentiate(value.(p[(:y, n-1)])[1], t, 2)([i])
      push!(x″, xₜ″)
      push!(y″, yₜ″)
      xₜ‴, yₜ‴ = differentiate(value.(p[(:x, n-1)])[1], t, 3)([i]), differentiate(value.(p[(:y, n-1)])[1], t, 3)([i])
      push!(x‴, xₜ‴)
      push!(y‴, yₜ‴)
    end
  end
  return model, solve_time.time, x, y, x′, y′, x″, y″, x‴, y‴
end

function scenario_1!()
  for obstacle in obstacles
    p = polygon(obstacle)
    poly!(p, color = "#CCCCCC")
  end
end

function plot!()
  app = WGLMakie.App() do session::WGLMakie.Session

  x, y = Observable([0.0]), Observable([0.0])
  x′, y′ = Observable([0.0]), Observable([0.0])
  x″, y″ = Observable([0.0]), Observable([0.0])
  x‴, y‴ = Observable([0.0]), Observable([0.0])
  λ = Observable(0.01)
  N = Observable(4) # number of trajectory segments
        
  h = Figure(; resolution=(900, 450))
  h_ax = Axis(h[1, :])
  
  scenario_1!()
  model, solve_time, x_λ, y_λ, x_λ′, y_λ′, x_λ″, y_λ″, x_λ‴, y_λ‴ = solve_for_optimal_path(λ[], N[])
  x[], y[], x′[], y′[], x″[], y″[], x‴[], y‴[] = x_λ, y_λ, x_λ′, y_λ′, x_λ″, y_λ″, x_λ‴, y_λ‴
  h_scatter = scatter!(h[1, :], x, y, color = :blue)
  
  f = Figure(; resolution=(450, 900))
  
  b_numSegments = Makie.Label(f[1, 1], "Number of Trajectory Segments")
  l_numSegments = Makie.Label(f[1, 2], string(N[]))
  b_numSegments_plus = Makie.Button(f[1, 3], label="+")
  on(b_numSegments_plus.clicks) do _
      N[] = N[] + 1
      l_numSegments.text[] = string(N[])
  end
  b_numSegments_minus = Makie.Button(f[1, 4], label="-")
  on(b_numSegments_minus.clicks) do _
      N[] = N[] - 1
      l_numSegments.text[] = string(N[])
  end
  b_lambda = Makie.Label(f[2, 1], "Lambda")
  l_lambda = Makie.Label(f[2, 2], string(λ[]))
  b_lambda_plus = Makie.Button(f[2, 3], label="+")
  on(b_lambda_plus.clicks) do _
      λ[] += 0.01
      l_lambda.text[] = string(λ[])
  end
  b_lambda_minus = Makie.Button(f[2, 4], label="-")
  on(b_lambda_minus.clicks) do _
      λ[] -= 0.01
      l_lambda.text[] = string(λ[])
  end
  
  b_startXpos = Makie.Label(f[3, 1], "Start x Position")
  l_startXpos = Makie.Label(f[3, 2], string(X₀[:x]))
  b_startXpos_plus = Makie.Button(f[3, 3], label="+")
  on(b_startXpos_plus.clicks) do _
      X₀[:x] += 1
      l_startXpos.text[] = string(X₀[:x])
  end
  b_startXpos_minus = Makie.Button(f[3, 4], label="-")
  on(b_startXpos_minus.clicks) do _
      X₀[:x] -= 1
      l_startXpos.text[] = string(X₀[:x])
  end
  b_startYpos = Makie.Label(f[4, 1], "Start y Position")
  l_startYpos = Makie.Label(f[4, 2], string(X₀[:y]))
  b_startYpos_plus = Makie.Button(f[4, 3], label="+")
  on(b_startYpos_plus.clicks) do _
      X₀[:y] += 1
      l_startYpos.text[] = string(X₀[:y])
  end
  b_startYpos_minus = Makie.Button(f[4, 4], label="-")
  on(b_startYpos_minus.clicks) do _
      X₀[:y] -= 1
      l_startYpos.text[] = string(X₀[:y])
  end
  b_endXpos = Makie.Label(f[5, 1], "End x Position")
  l_endXpos = Makie.Label(f[5, 2], string(X₁[:x]))
  b_endXpos_plus = Makie.Button(f[5, 3], label="+")
  on(b_endXpos_plus.clicks) do _
      X₁[:x] += 1
      l_endXpos.text[] = string(X₁[:x])
  end
  b_endXpos_minus = Makie.Button(f[5, 4], label="-")
  on(b_endXpos_minus.clicks) do _
      X₁[:x] -= 1
      l_endXpos.text[] = string(X₁[:x])
  end
  b_endYpos = Makie.Label(f[6, 1], "End y Position")
  l_endYpos = Makie.Label(f[6, 2], string(X₁[:y]))
  b_endYpos_plus = Makie.Button(f[6, 3], label="+")
  on(b_endYpos_plus.clicks) do _
      X₁[:y] += 1
      l_endYpos.text[] = string(X₁[:y])
  end
  b_endYpos_minus = Makie.Button(f[6, 4], label="-")
  on(b_endYpos_minus.clicks) do _
      X₁[:y] -= 1
      l_endYpos.text[] = string(X₁[:y])
  end
      
  b_startXvel = Makie.Label(f[7, 1], "Start x Velocity")
  l_startXvel = Makie.Label(f[7, 2], string(X₀′[:x]))
  b_startXvel_plus = Makie.Button(f[7, 3], label="+")
  on(b_startXvel_plus.clicks) do _
      X₀′[:x] += 1
      l_startXvel.text[] = string(X₀′[:x])
  end
  b_startXvel_minus = Makie.Button(f[7, 4], label="-")
  on(b_startXvel_minus.clicks) do _
      X₀′[:x] -= 1
      l_startXvel.text[] = string(X₀′[:x])
  end
  b_startYvel = Makie.Label(f[8, 1], "Start y Velocity")
  l_startYvel = Makie.Label(f[8, 2], string(X₀′[:y]))
  b_startYvel_plus = Makie.Button(f[8, 3], label="+")
  on(b_startYvel_plus.clicks) do _
      X₀′[:y] += 1
      l_startYvel.text[] = string(X₀′[:y])
  end
  b_startYvel_minus = Makie.Button(f[8, 4], label="-")
  on(b_startYvel_minus.clicks) do _
      X₀′[:y] -= 1
      l_startYvel.text[] = string(X₀′[:y])
  end
  b_endXvel = Makie.Label(f[9, 1], "End x Velocity")
  l_endXvel = Makie.Label(f[9, 2], string(X₁′[:x]))
  b_endXvel_plus = Makie.Button(f[9, 3], label="+")
  on(b_endXvel_plus.clicks) do _
      X₁′[:x] += 1
      l_endXvel.text[] = string(X₁′[:x])
  end
  b_endXvel_minus = Makie.Button(f[9, 4], label="-")
  on(b_endXvel_minus.clicks) do _
      X₁′[:x] -= 1
      l_endXvel.text[] = string(X₁′[:x])
  end
  b_endYvel = Makie.Label(f[10, 1], "End y Velocity")
  l_endYvel = Makie.Label(f[10, 2], string(X₁′[:y]))
  b_endYvel_plus = Makie.Button(f[10, 3], label="+")
  on(b_endYvel_plus.clicks) do _
      X₁′[:y] += 1
      l_endYvel.text[] = string(X₁′[:y])
  end
  b_endYvel_minus = Makie.Button(f[10, 4], label="-")
  on(b_endYvel_minus.clicks) do _
      X₁′[:y] -= 1
      l_endYvel.text[] = string(X₁′[:y])
  end
      
  b_startXacc = Makie.Label(f[11, 1], "Start x Acceleration")
  l_startXacc = Makie.Label(f[11, 2], string(X₀″[:x]))
  b_startXacc_plus = Makie.Button(f[11, 3], label="+")
  on(b_startXacc_plus.clicks) do _
      X₀″[:x] += 1
      l_startXacc.text[] = string(X₀″[:x])
  end
  b_startXacc_minus = Makie.Button(f[11, 4], label="-")
  on(b_startXacc_minus.clicks) do _
      X₀″[:x] -= 1
      l_startXacc.text[] = string(X₀″[:x])
  end
  b_startYacc = Makie.Label(f[12, 1], "Start y Acceleration")
  l_startYacc = Makie.Label(f[12, 2], string(X₀″[:y]))
  b_startYacc_plus = Makie.Button(f[12, 3], label="+")
  on(b_startYacc_plus.clicks) do _
      X₀″[:y] += 1
      l_startYacc.text[] = string(X₀″[:y])
  end
  b_startYacc_minus = Makie.Button(f[12, 4], label="-")
  on(b_startYacc_minus.clicks) do _
      X₀″[:y] -= 1
      l_startYacc.text[] = string(X₀″[:y])
  end
  b_endXacc = Makie.Label(f[13, 1], "End x Acceleration")
  l_endXacc = Makie.Label(f[13, 2], string(X₁″[:x]))
  b_endXacc_plus = Makie.Button(f[13, 3], label="+")
  on(b_endXacc_plus.clicks) do _
      X₁″[:x] += 1
      l_endXacc.text[] = string(X₁″[:x])
  end
  b_endXacc_minus = Makie.Button(f[13, 4], label="-")
  on(b_endXacc_minus.clicks) do _
      X₁″[:x] -= 1
      l_endXacc.text[] = string(X₁″[:x])
  end
  b_endYacc = Makie.Label(f[14, 1], "End y Acceleration")
  l_endYacc = Makie.Label(f[14, 2], string(X₁″[:y]))
  b_endYacc_plus = Makie.Button(f[14, 3], label="+")
  on(b_endYacc_plus.clicks) do _
      X₁″[:y] += 1
      l_endYacc.text[] = string(X₁″[:y])
  end
  b_endYacc_minus = Makie.Button(f[14, 4], label="-")
  on(b_endYacc_minus.clicks) do _
      X₁″[:y] -= 1
      l_endYacc.text[] = string(X₁″[:y])
  end
  i = Figure(; resolution=(450, 50))
  Makie.Label(i[1, 1], "Solver status:")
  l_state = Makie.Label(i[1, 2], "Idle")
  Makie.Label(i[2, 1], "Solve time (seconds):")
  l_time = Makie.Label(i[2, 2], "")
        
  g = Figure(; resolution=(900, 900))
  # plot displacement
  ax = Axis(g[1, 1], xlabel="time", ylabel="x position", title="Trajectory x position over time")
  u = Observable([0])
  u[] = Vector(1:length(x[]))
  lines!(ax, u, x)
  ax = Axis(g[1, 2], xlabel="time", ylabel="y position", title="Trajectory y position over time")
  lines!(ax, u, y)

  # plot velocity
  ax = Axis(g[2, 1], xlabel="time", ylabel="x velocity", title="Trajectory x velocity over time")
  lines!(ax, u, x′)
  ax = Axis(g[2, 2], xlabel="time", ylabel="y velocity", title="Trajectory y velocity over time")
  lines!(ax, u, y′)

  # plot acceleration
  ax = Axis(g[3, 1], xlabel="time", ylabel="x acceleration", title="Trajectory x acceleration over time")
  lines!(ax, u, x″)
  ax = Axis(g[3, 2], xlabel="time", ylabel="y acceleration", title="Trajectory y acceleration over time")
  lines!(ax, u, y″)

  # plot jerk
  ax = Axis(g[4, 1], xlabel="time", ylabel="x jerk", title="Trajectory x jerk over time")
  lines!(ax, u, x‴)
  ax = Axis(g[4, 2], xlabel="time", ylabel="y jerk", title="Trajectory y jerk over time")
  lines!(ax, u, y‴)

  updatebtn = JSServe.Button("update")
  on(updatebtn) do _
    l_state.text[] = "Running..."
    model, solve_time, x_λ, y_λ, x_λ′, y_λ′, x_λ″, y_λ″, x_λ‴, y_λ‴ = solve_for_optimal_path(λ[], N[])
    if termination_status(model) != OPTIMAL
      l_state.text[] = "Failed to produce an optimal solution. Try again."
      l_time.text[] = ""
    else
      l_state.text[] = "Solution is OPTIMAL."
      l_time.text[] = string(solve_time) * " seconds"
      x.val = x_λ
      y.val = y_λ
      u.val = 1:length(x_λ)
      x′.val = x_λ′
      y′.val = y_λ′
      x″.val = x_λ″
      y″.val = y_λ″
      x‴.val = x_λ‴
      y‴.val = y_λ‴
      x[], y[], x′[], y′[], x″[], y″[], x‴[], y‴[] = x_λ, y_λ, x_λ′, y_λ′, x_λ″, y_λ″, x_λ‴, y_λ‴
      u[] = Vector(1:length(x_λ))
    end
  end

  return DOM.div(DOM.div(DOM.h1("Optimal path planning with polynomial based trajectories and polyhedral constraints"),
    DOM.p("Last updated: 31 Dec 2023"),
    DOM.p("This app (written entirely in Julia) demonstrates the use of a conic and MIPS solver to optimize for the smoothest trajectory through a set of polyhedral constraints using Sum of Squares, Big M method and 2nd order polynomial basis functions.")),
    DOM.p("P.S. The polynomial order used is crucial for a feasible solution and to avoid numerical stability issues using the solver."),
    DOM.p("There are at least three segments making up the path since each segment can belong to only one of three regions, however there can be multiple segments within a region."),
    DOM.p("The regions can be thought of as safe fly zones or collision free zones which are defined by the gaps between obstacles shown in gray."),
    DOM.p("You can try several different start and end positions and the solver will work out an optimal path. No path means no feasible solution can be found or that the solver timed out."),
    DOM.p("The implementation is derived using the Pajarito with Hypatia and MOSEK solvers and the work done by Huchette, Vielma and others."),
    DOM.p("An example for where this could be used is robotic motion control."),
    DOM.p("For more information, a good book on this topic is Semidefinite Optimization and Convex Algebraic Geometry by Blekherman, Parrilo and Thomas."),
    DOM.p("This app has been updated with my latest research which adds a lambda parameter to tradeoff between a trajectory's smoothness and obstacle proximity."),
    DOM.p("As you can see it doesn't do a particularly great job in some spots since it struggles to go around the obstacle."),
    DOM.p("A way to mitigate this is to manually assign segments to safe regions but this comes at the expense of the MIPS aspect of the solution."),
    DOM.p("The effect of this alternate approach will be shown in a future release."),
    DOM.div(DOM.div(style="float:left; text-align:center", f, updatebtn)),
    DOM.div(i),
    DOM.div(h),
    DOM.div(g),
    DOM.p(style="text-align:center", "Alexander Leong 2023"))
  end

  server = JSServe.Server(app, "0.0.0.0", 8080)
  JSServe.HTTPServer.start(server)
  wait(server)
end
  
plot!()