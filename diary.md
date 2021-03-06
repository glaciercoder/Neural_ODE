# 暑研工作日志

​																													——Started from July 1th

基本完成了春季学期所有大作业，可以正式开始工作了。





## Day1

### 学习Julia基本概念与管理

[REPL](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop)

包管理 [Pkg](https://docs.julialang.org/en/v1/stdlib/Pkg/)

[Julia Environment and notebook](https://towardsdatascience.com/how-to-setup-project-environments-in-julia-ec8ae73afe9c)

阅读文献

短期工作目标：读懂示例代码func.jl

（MNIST变量太多，ODE solver算不动）



## Day 2

### 学习Julia基本语法

#### 运算

Vectorized dot：可用于单元，二元运算符，可用于比较，还可以用于函数

```julia
#Always use vec dot is tedious, use macro @. to vectorize all operations
Y = [1,2,3]
X = similar(Y)
@. X = Y^2
```

#### 函数

函数定义

```julia
# funciton can be defined without return or with return
function f(x,y)
  x+y
end			
# Anonymous function is widely used in the map function
map(x->x^2+2x-1,[1,2,3])
# Anonymous function can receive multiple arguements
(x,y,z)->2x+y-z
# Function in Julia can return multiple values in tuple form
function foo(a,b) 
  a+b, a*b
end
x,y = foo(a,b)

#Do syntax is a useful feature in function with function as arguments
map(f(x),[1,2,3])
# can be written as
map([1,2,3]) do
  f(x)
end
#Function in Julia can be composed as in math
(sqrt ∘ +)(3, 6)
#Function can also be piped as in shell 
1:10 |> sum |> sqrt
#An useful example
[1:5;] .|> [x->x^2, inv, x->2*x, -, isodd]
```

#### 控制流

```julia
# Compound expreesion using begin blocks
z = begin 
  x=1 
  y=2 
  x+y
end
# Compound expreesion using ; chain
z = (x = 1; y = 2; x + y)
# 'If' expression is leaky and can not use non-boolean value to judge
# A loop can use multiple variables
for i = 1:2, j = 3:4 
  println((i, j))
end
# can alse use
for (j, k) in zip([1 2 3], [4 5 6 7]) 
  println((j,k))
end

```

#### Type

```julia
# :: can be used to make type assertion
(1+2)::Int
# Subtype 
<:
# Abstract type is just like virtual base class, it can not be instantiated.
# Composite type can be defined using struct, type assertion can be used optionally
struct Foo
	A
	B::Int
	C
end
# Use fielsname() to check the list of the fieldsname
# Use . to access the member of the struct
# The type in Julia is a kind of generic programming. Parametric Composite types
struct point{T}
	x::T
	y::T
end
# Combining parametric type and subtype, we can write generic function
function norm(p::Point{<:Real}) 
  sqrt(p.x^2 + p.y^2)
end
# or write as
function norm(p::Point{T} where T<:Real)

```

#### Method and Dispatch(重载)

```julia
# Using :: constrain the type of the argument of a function
f(x::Float64, y::Float64) = 2x + y
# define a function multiple times with different numbers of argument of different argument types can generate multiple methods, methods() function is useful to see the methods of a function
methods(f)
# Parametric Methods is a special usage 
myappend(v::Vector{T}, x::T) where {T} = [v..., x]
# Any newly-defined method will not take effect in the current environment


```



### 学习Flux

$$
f(\vec{x},\vec{y}) = (x_1-y_1)^{2}+ (x_2-y_2)^{2}+\cdots+(x_n-y_n)^{2}\\
\nabla f = [2(x_1-y_1),2(x_2-y_2),\dots,2(x_n-y_n),-2(x_1-y_1),\dots,-2(x_n-y_n)]\\
\nabla f = [\frac{\partial f}{\partial \vec{x}},\frac{\partial f}{\partial \vec{y}}]
$$



```julia
# The most important operation in Flux is gradient
# For a monovariable function
f(x) = 3x^2 + 2x + 1;
df(x) = gradient(f, x)[1]; # df/dx = 6x + 2
# For muli-variable function, gradient returns its gradient vector for each vector variable respectively
f(x, y) = sum((x .- y).^2);
gradient(f, [2, 1], [2, 0])
([0, 2], [0, -2])
# Use params will be more convenient
gradient(f(x,y),params(x,y))
# Use do syntax, gradient can also be used as 
gs = gradient(params(x,y)) do
  f(x,y)
end
g[x] = ...
g[y] = ...


```



## Day 3

在出差，一天没有干活



## Day 4

#### 继续学习Flux和Julia

[W3CSchool](https://www.w3cschool.cn/julia/ik8o1jfg.html)

#### Julia Array

```julia
#execute the external file
include("file.jl")
# Chain comparison
1 < 2 <= 2 < 3 == 3 > 2 >= 1 == 1 < 3 != 5
# Array operations
rand(dims) # uniform [0,1] vector
randn(dims) # Guassion N(0,1) vector
eye(n) # Unit
# Connection
# hcat() make a matrix
# number of rows of each array must match
# vcat()  make a vector
#Attention to the difference between vector and Matrix
# Matrix
A = [1 2;1 2]
# Vector
B = [1,2,3,4]
# The following  expressons are the same
vcat(1,2,3)
vcat(1,2,3...)
vcat(1:3)
vcat(1:3...)
# The following two expressions are different
hcat(0:5) # get a 6x1 matrix
hcat(0:5...) # get a 1x6 matrix
# Broadcast function is important

```



 In Flux, *models are conceptually predictive functions*, nothing more

```julia
# The first important function in Flus is model use
model = Dense(1,1)
# For loss function, use
losses(x,y) = Flux.Losses.mse(predict(x), y)
# params will get all parameters that will be changed in the model
parameters = params(model)
# Choose optimiser
opt = Descent()
Descent(0.1)
# When we have data, model, optimiser, loss function
 using Flux: train!
train!(loss, parameters, data, opt)ß

# Connect the layer
model2 = Chain(
  Dense(10, 5, σ),
  Dense(5, 2),
  softmax)


```



## Day 5

### 学习DifferentailEquations.jl

#### Define a problem

A problem contains `equation` ,`inital condition`, `time span`(namely, integral interval)

```julia
\frac{du}{dt} = f(u,p,t)
prob = ODEProblem(f,u0,tspan,p)
# p is the parameter, the initial conditino can be written as the function of p
u0(p,t0)
# So is the timespan
tspan(p)
# Anonymous function can be used
prob = ODEProblem((u,p,t)->u,(p,t0)->p[1],(p)->(0.0,p[2]),(2.0,1.0))
using Distributions
prob = ODEProblem((u,p,t)->u,(p,t)->Normal(p,1),(0.0,1.0),1.0)
```

#### Solve the problem

```julia
# The solver use a specified alogrithm, and some arguements to solve the problem
sol = solve(prob,alg;kargs)
```

The solution has different fields :

`u `: the Vector of values at each timestep

`t`:  the times of each timestep

```julia
sol[j] # access value of timestp j
sol.t[j] # access jth timestep
sol[i,j] # i component at jth timestep
sol[i,:] # all time series
# The sol interface supports interpolation to get the value at any time
sol(t,deriv=Val{0};idxs=nothing,continuity=:left)
# The sol supports comprehension
[t+2u for (u,t) in tuples(sol)]
[t+3u-2du for (u,t,du) in zip(sol.u,sol.t,sol.du)]
```

#### Analyze the solution

Since the solution interface is unified in Julia, general method can be used to analyze the solution. The common used tool is Plots.jl

```julia
using Plots
plot(sol)
savefig("My_plot.png")
plot(sol,plotdensity=1000) # points used to plot can be set
plot(sol,tspan = (0.0,4.0)) # choose the plot timespan
# plot many functions at a time
vars = [(f1,0,1), (f2,1,3), (f3,4,5)] # where 0,1,3,4,5 are different vars, 0 for time
# plot var2 vs var1
vars = [(1,2)]
# if var1 is time, 1 can be omitted
vars = [2]
# the vector will be expand to general form
vars = [(1, 2, 3), (4, 5, 6)]
# is eual to
vars = [(1,4), (2,5), (3,6)]

```

```julia
# An example
using DifferentialEquations, Plots
function lorenz(du,u,p,t)
 du[1] = p[1]*(u[2]-u[1])
 du[2] = u[1]*(p[2]-u[3]) - u[2]
 du[3] = u[1]*u[2] - p[3]*u[3]
end

u0 = [1., 5., 10.]
tspan = (0., 100.)
p = (10.0,28.0,8/3)
prob = ODEProblem(lorenz, u0, tspan,p)
sol = solve(prob)
xyzt = plot(sol, plotdensity=10000,lw=1.5)
xy = plot(sol, plotdensity=10000, vars=(1,2))
xz = plot(sol, plotdensity=10000, vars=(1,3))
yz = plot(sol, plotdensity=10000, vars=(2,3))
xyz = plot(sol, plotdensity=10000, vars=(1,2,3))
plot(plot(xyzt,xyz),plot(xy, xz, yz, layout=(1,3),w=1), layout=(2,1))
f(t,x,y,z) = (t,sqrt(x^2+y^2+z^2))
plot(sol,vars=(f,0,1,2,3))
```

## Day 6

#### 继续学习Julia

- By convention, function names ending with an exclamation point (`!`) modify their arguments. Some functions have both modifying (e.g., `sort!`) and non-modifying (`sort`) versions.

- ```julia
  @show # a Macro to print the value of variables
  ```

- Radom: Julia uses Random Module for random number 

  ```julia
  # The most general form for the generation of random number
  rand([rng=GLOBAL_RNG], [S], [dims...]) 
  #where RNG is MersenneTwister by default
  # S defaults to Float64
  # Attention :
  rand((2,3)) #means choose a number from 2 and 3, (2,3) is taken as S
  rand(Floats,(2,3) # Generates 2x3 array
  # another form is
  rand!([rng=GLOBAL_RNG], A, [S=eltype(A)])
  rand(dims) # uniform [0,1] vector
  randn(dims) # Guassion N(0,1) vector
  rand(a:b,c,d) # cxd dimensions
  ```

- ```
  # In notebook, use 
  ?[fun_name] # to get the usage of the function
  Array{Float64, 3} # 3 for the dimension, not numbers
  ```

- 



## Day 7

讨论了暑研的方向，打算更改方向，去搞反应合成

Physics informed ml

机器学习 反应路径

两篇文章的区别：应用场景

美国biometric比较发达 ，MIT ME一般人的都在搞这个

合作者是大牛

模型生成数据已经完成

下一步打算使用真实数据学习

数据处理之类的困难重重

超前于工业界

数据的来源：已经拿到 维度要高很多 噪声很大 无groundtruth

看相关的论文

记录代码

Test_beeline. HSC config

基于他们之前的一篇文章
https://arxiv.org/pdf/2104.06467.pdf
是从数据中推断细胞/分子反应体系的反应路径及参数，现在有新的实验数据，可以把这部分工作做进一步的开展完善
https://github.com/jiweiqi/CellBox.jl



[coursera](https://www.coursera.org/learn/julia-programming/lecture/4D0rQ/functions-in-julia)



## Day 8，9

安装使用atom

阅读cellbox 代码

建议下次一边跑block一边读，效率会提高不少 using ## to make blocks in Atom, and option+enter to execute



#### header.jl

```julia
### Parsing config
using ArgParse
using YAML

s = ArgParseSettings()
@add_arg_table! s begin
    "--disable-display"
        help = "Use for UNIX server"
        action = :store_true
    "--expr_name"
        help = "Define expr name"
        required = false
    "--is-restart"
        help = "Continue training?"
        action = :store_true
end
parsed_args = parse_args(ARGS, s)

if parsed_args["expr_name"] != nothing
    expr_name = parsed_args["expr_name"]
    is_restart = parsed_args["is-restart"]
else
    runtime = YAML.load_file("./runtime.yaml")
    expr_name = runtime["expr_name"]
    is_restart = Bool(runtime["is_restart"])
end
conf = YAML.load_file("$expr_name/config.yaml")
```

[ArgParse](https://argparsejl.readthedocs.io/en/latest/argparse.html)

[What is argument/option/flag](https://unix.stackexchange.com/questions/285575/whats-the-difference-between-a-flag-an-option-and-an-argument)

[More on ArgParse](https://docs.python.org/3/howto/argparse.html)

[YAML](https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started)

```julia
cd(dirname(@__DIR__))
ENV["GKSwstype"] = "100"
fig_path = string("./results/", expr_name, "/figs")
ckpt_path = string("./results/", expr_name, "/checkpoint")
config_path = "./results/$expr_name/config.yaml"
pyplot()

if is_restart
    println("Continue to run $expr_name ...\n")
else
    println("Runing $expr_name ...\n")
end

fig_path = string(expr_name, "/figs")
ckpt_path = string(expr_name, "/checkpoint")

if !is_restart
    if ispath(fig_path)
        rm(fig_path, recursive=true)
    end
    if ispath(ckpt_path)
        rm(ckpt_path, recursive=true)
    end
end

if ispath(fig_path) == false
    mkdir(fig_path)
    mkdir(string(fig_path, "/conditions"))
end

if ispath(ckpt_path) == false
    mkdir(ckpt_path)
end

if haskey(conf, "grad_max")
    grad_max = conf["grad_max"]
else
    grad_max = Inf
end

ns = Int64(conf["ns"]); # number of nodes / species
tfinal = Float64(conf["tfinal"]);
ntotal = Int64(conf["ntotal"]);  # number of samples for each perturbation
nplot = Int64(conf["nplot"]);
batch_size = Int64(conf["batch_size"]);  # STEER

n_exp_train = Int64(conf["n_exp_train"]);
n_exp_val = Int64(conf["n_exp_val"]);
n_exp_test = Int64(conf["n_exp_test"]);

n_exp = n_exp_train + n_exp_val + n_exp_test;
noise = Float64(conf["noise"])
opt = ADAMW(Float64(conf["lr"]), (0.9, 0.999), Float64(conf["weight_decay"]));

n_iter_max = Int64(conf["n_iter_max"])
n_plot = Int64(conf["n_plot"]);
n_iter_buffer = Int64(conf["n_iter_buffer"])
n_iter_burnin = Int64(conf["n_iter_burnin"])
n_iter_tol = Int64(conf["n_iter_tol"])
convergence_tol = Float64(conf["convergence_tol"])
```

[Julia Fileysystem](https://docs.julialang.org/en/v1/base/file/)

设置gradmax是为了防止梯度爆炸



#### network.jl

```julia
if "data" in keys(conf)
    idx_order = randperm(n_exp) # randperm -> generate a random pertubation of n_exp
    pert = DataFrame(CSV.File(string(conf["data"],"/pert.csv"); header=false))
    μ_list = convert(Matrix,pert)[idx_order,:] # don't understand
else
    μ_list = randomLHC(n_exp, ns) ./ n_exp;
    nμ = Int64(conf["n_mu"])
    for i = 1:n_exp
        nonzeros = findall(μ_list[i, :].>0)
        ind_zero = sample(nonzeros, max(0, length(nonzeros)-nμ), replace=false)
        μ_list[i, ind_zero] .= 0
    end
end
```

这段的意义是控制变量，ns表示干扰维数，number of nodes，每次控制只有一个维度上的干扰起作用

[CSV and DataFrame](https://towardsdatascience.com/read-csv-to-data-frame-in-julia-programming-lang-77f3d0081c14)

```julia
function gen_network(m; weight_params=(-1., 1.), sparsity=0., drop_range=(-1e-1, 1e-1))

    # uniform random for W matrix
    w = rand(Uniform(weight_params[1], weight_params[2]), (m, m))

    # Drop small values
    @inbounds for i in eachindex(w)
        w[i] = ifelse(drop_range[1]<=w[i]<=drop_range[2], 0, w[i])
    end

    # Add sparsity
    p = [sparsity, 1 - sparsity]
    w .*= sample([0, 1], weights(p), (m, m), replace=true)

    # Add α vector
    α = abs.(rand(Uniform(weight_params[1], weight_params[2]), (m))) .+ 0.5

    return hcat(α, w)
end
```

增加稀疏性

[@inbounds](https://docs.julialang.org/en/v1/devdocs/boundscheck/)

ifelse function -----see document 



```julia
if "network" in keys(conf)
    df = DataFrame(CSV.File(conf["network"]))
    nodes = names(df)
    w = convert(Matrix, df[:,2:end])
    @assert size(w)[1] == size(w)[2]
    @assert size(w)[1] == ns
    if "randomize_network" in keys(conf)
        w_rand = rand(Normal(1, conf["randomize_network"]), (ns, ns))
        w = w .* w_rand
    end
    if "alpha" in keys(conf)
        α = ones(ns) .* conf["alpha"]
    else
        α = ones(ns) .* 0.2
    end
    p_gold = hcat(α, w)
elseif "data" in keys(conf)
    p_gold = hcat(ones(ns),zeros(ns,ns))
else
    p_gold = gen_network(ns, weight_params=(-1.0, 1.0),
                         sparsity=Float64(conf["sparsity"]),
                         drop_range=(Float64(conf["drop_range"]["lb"]), Float64(conf["drop_range"]["ub"])));
end
p = gen_network(ns; weight_params=(0.0, 0.01), sparsity=0);
# show_network(p)
```

这一段的convert不work

[解决方法](https://discourse.julialang.org/t/cannot-convert-an-object-of-type-dataframe-to-an-object-of-type-array/61598/2)

　

[Macros in Julia](https://stackoverflow.com/questions/58137512/why-use-macros-in-julia)

```julia
function show_network(p)
    println("p_gold")
    show(stdout, "text/plain", round.(p_gold, digits=2))
    println("\np_learned")
    show(stdout, "text/plain", round.(p, digits=2))
end

function loss_network(p)
     # distalpha = cosine_dist(p_gold[:,1],p[:,1])
     # distw = cosine_dist(p_gold[:,2:end],p[:,2:end])
     @inbounds coralpha = cor(p_gold[:,1],p[:,1])
     @inbounds corw = cor([p_gold[:,2:end]...],[p[:,2:end]...])
     return coralpha, corw
end

function cellbox!(du, u, p, t)
    @inbounds du .= @view(p[:, 1]) .* tanh.(@view(p[:, 2:end]) * u - μ) .- u
end

tspan = (0, tfinal);
ts = 0:tspan[2]/ntotal:tspan[2];
ts = ts[2:end];
u0 = zeros(ns);
prob = ODEProblem(cellbox!, u0, tspan, saveat=ts);

function max_min(ode_data)
    return maximum(ode_data, dims=2) .- minimum(ode_data, dims=2)
end

if "data" in keys(conf)
    expr = DataFrame(CSV.File(string(conf["data"],"/expr.csv"); header=false))
    ode_data_list = zeros(Float64, (n_exp, ns, ntotal));
    ode_data_list[:,:,1] = convert(Matrix,expr)[idx_order,:]
else
    ode_data_list = zeros(Float64, (n_exp, ns, ntotal));
    yscale_list = [];
    for i = 1:n_exp
        global μ = μ_list[i, 1:ns]
        ode_data = Array(solve(prob, Tsit5(), u0=u0, p=p_gold))

        ode_data += randn(size(ode_data)) .* noise
        ode_data_list[i, :, :] = ode_data

        push!(yscale_list, max_min(ode_data))
    end
    yscale = maximum(hcat(yscale_list...), dims=2);
end
```

[@view](https://discourse.julialang.org/t/could-you-explain-what-are-views/17535/2)

[push!](https://discourse.julialang.org/t/using-push/30935)



```julia
function predict_neuralode(u0, p, i_exp=1, batch=ntotal, saveat=true)
    global μ = μ_list[i_exp, 1:ns]
    if saveat
        @inbounds _prob = remake(prob, p=p, tspan=[0, ts[batch]])
        pred = Array(solve(_prob, Tsit5(), saveat=ts[1:batch],
                     sensealg=InterpolatingAdjoint()))
    else # for full trajectory plotting
        @inbounds _prob = remake(prob, p=p, tspan=[0, ts[end]])
        pred = Array(solve(_prob, Tsit5(), saveat=0:ts[end]/nplot:ts[end]))
    end
    return pred
end
predict_neuralode(u0, p, 1);
```

What is saveat

gold是啥的缩写

saveat是啥的缩写



 ## Day 10,11,12

外出装车，没有干活



## Day 13

开会：

用一个点

conditions validation test data

有历史数据，有烟花过程

原始数据99维，24为简化过后的

24 5-10min训完

设置画图频率，以及步数，可以提高训练速度

[arrays](https://www.cnblogs.com/kirito-c/p/10268441.html)

所有数据的噪声基本是一个量级的，对于过于小的实验值，这个噪声会让loss爆炸，所以要scale一下



## Day 14

[Think Julia](https://benlauwens.github.io/ThinkJulia.jl/latest/book.html)



咕咕咕了好几天

## Day 15

继续看代码和paper

**Here's a secret: Plots.jl isn't actually a plotting package! Plots.jl is a plotting metapackage: it's an interface over many different plotting libraries. Thus what Plots.jl is actually doing is interpreting your commands and then generating the plots using another plotting library. This plotting library in the background is referred to as the **backend**. *



## Day 16

开会，讨论了下一步进行的方向

1. 用现在的code
2. 随机微分方程（SDE）

现有方法：跑很多遍，做BP

对weight随机取样，训网络，输入为参数，额外加上时间t和pertubation，输出对应的pertubation，和t下的protein level，预测均值和方差，和实验的均值和方差，代码实现方便，不需要ODE积分器（能否找到更adapative），

end to end

SDE(使用合成数据)

没有real data

先用ODE做，然后转SDE

先用ODE实现

先用小规模数据

可以用SDE生成数据，最终学到的还是matrix

用内置的extrema函数求scale

离散化或者scale down



## Day17

读Net2Net代码，直接从数据到输出的黑盒模型，但是把t作为一个维度的输入，得到在任意时刻的状态量，这样的好处是不用进行积分，可以大大减小计算量，缺点是丢失了可解释性。

Resnet

The skip connections in ResNet solve the problem of vanishing gradient in deep neural networks by allowing this alternate shortcut path for the gradient to flow through. The other way that these connections help is by allowing the model to learn the identity functions which ensures that the higher layer will perform at least as good as the lower layer, and not worse. 











