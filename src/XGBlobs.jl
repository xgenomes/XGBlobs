module XGBlobs

using LinearAlgebra
using OffsetArrays
using StaticArrays
using SparseArrays
using Distributions

# include("psf.jl")

export GaussianKernel, all_modes, KDE, integrate, render, FixedRadiusCellList
export inner_product, inner_product_offset_gradient

#### Cell lists. Super simple...
@inline _bin_idx(x :: Float64, bin_width :: Float64) = ceil(Int64, x/bin_width)

struct FixedRadiusCellList # This code is shared with Drifter, should share, but use the OffsetArray version
    cells :: Vector{Vector{SVector{2, Float64}}}
    radius :: Float64
    indexes :: OffsetMatrix{Int, SparseMatrixCSC{Int, Int}}
    function FixedRadiusCellList(r, maxx, maxy)
      _maxx = _bin_idx(maxx, r) * 2
      _maxy = _bin_idx(maxy, r) * 2
      new(Vector{SVector{2, Float64}}[], r, OffsetArray(spzeros(Int, _maxx * 2 + 1, _maxy * 2 + 1), -maxx:maxx, -maxy:maxy))
  end
end

function Base.push!(t :: FixedRadiusCellList, p :: SVector{2, Float64})
    k = _bin_idx.(p, t.radius)

    list = if t.indexes[k[1], k[2]] == 0
        t.indexes[k[1], k[2]] = length(t.cells) + 1
        push!(t.cells, SVector{2, Float64}[])
        t.cells[end]
    else
        t.cells[t.indexes[k[1], k[2]]]
    end

    push!(list, p)
end

function FixedRadiusCellList(points :: Vector{SVector{2, Float64}}, radius :: Float64, maxx :: Float64, maxy :: Float64)
    t = FixedRadiusCellList(radius, maxx, maxy)
    for p in points
        push!(t, p)
    end
    t
end

function Base.haskey(t :: FixedRadiusCellList, k :: NTuple{2, Int})
  getindex(t.indexes, k...) > 0
end

function Base.getindex(t :: FixedRadiusCellList, k :: NTuple{2, Int})
  t.cells[getindex(t.indexes, k...)]
end

function foldl_range_query(op,init, t :: FixedRadiusCellList, p :: SVector{2, Float64})
    offsets = (-1, 0, 1)
    i_x = _bin_idx(p[1], t.radius)
    i_y = _bin_idx(p[2], t.radius)
    r_sq = t.radius*t.radius

    for o_x in offsets, o_y in offsets
        k = (i_x + o_x, i_y + o_y)
        if haskey(t, k)
          for n in t[k]
              d = p .- n
              if dot(d, d) ≤ r_sq
                  init = op(n, init)
              end
          end
        end
    end
    init
end


function has_neighbors(t :: FixedRadiusCellList, p :: SVector{2, Float64}, radius :: Float64)
    r_sq = radius*radius
    offsets = (-1, 0, 1)

    i_x = _bin_idx(p[1], t.radius)
    i_y = _bin_idx(p[2], t.radius)

    for o_x in offsets, o_y in offsets
        k = (i_x + o_x, i_y + o_y)
        if haskey(t, k)
          for n in t[k]
              d = p .- n
              if dot(d, d) ≤ r_sq
                  return true
              end
          end
        end
    end
    return false
end

struct GaussianKernel
  σ :: Float64
  precision :: Float64
  mul :: Float64
  GaussianKernel(σ) = new(σ, 1.0/(σ^2), (1.0/(2pi*σ^2)))
end

@inline (g :: GaussianKernel)(z :: Float64) = g.mul*exp(-g.precision*z)

@inline function taylor(g :: GaussianKernel, z :: Float64)
  k = exp(-g.precision*z)
  (g.mul*k, -g.mul*g.precision*k, g.mul*g.precision*g.precision*k)
end

struct KDE
  K :: GaussianKernel
  T :: FixedRadiusCellList
  points :: Vector{SVector{2, Float64}}
end

function KDE(σ :: Float64, points :: Vector{SVector{2, Float64}}, radius = 4*σ)
  maxx = maximum(abs(p[1]) for p ∈ points)
  maxy = maximum(abs(p[2]) for p ∈ points)
  KDE(GaussianKernel(σ), FixedRadiusCellList(points, radius, maxx, maxy), points)
end

struct KDEEvaluator
  p :: SVector{2,Float64}
  K :: GaussianKernel
end

@inline function (s :: KDEEvaluator)(n :: SVector{2, Float64}, state :: Float64)
  d = n .- s.p
  state + s.K(0.5*dot(d, d))
end

@inline (ρ :: KDE)(p :: SVector{2, Float64}) = foldl_range_query(KDEEvaluator(p, ρ.K), 0.0, ρ.T, p)
@inline (ρ :: KDE)(x :: Float64, y :: Float64) = ρ(SVector(x,y))
@inline (ρ :: KDE)(v :: Vector{Float64}) = ρ(v[1], v[2])

struct KDETaylorEvaluator
  p :: SVector{2,Float64}
  K :: GaussianKernel
end

@inline function (k :: KDETaylorEvaluator)(n, (v, g, H))
  d = k.p .- n
  r_sq = 0.5*dot(d, d)
  (K, K_p, K_pp)= taylor(k.K, r_sq)
  (v + K, g + d*K_p, H + K_pp*d*d' + K_p*I)
end

taylor(ρ :: KDE, x) = foldl_range_query(KDETaylorEvaluator(x,ρ.K), (0.0, SVector(0.0,0.0), SMatrix{2,2}(0.0,0.0,0.0,0.0)), ρ.T, x)

function _newton(ρ :: KDE, x, radius, max_iters, min_v)
  v = 0.0

  x_zero = x
  r_sq = radius*radius
  for i in 1:max_iters
    (v, g, H) = taylor(ρ, x)
    d = x-x_zero
    if v < min_v || dot(d, d) > r_sq
      return x, v, false
    end
    lam, U = eigen(Hermitian(H))

    if lam[1] > 0.0 || lam[2] > 0.0
      return x, v, false
    end
    if norm(g) < 1E-8
      break
    end
    delta = H\g
    x = x - delta
  end
  x, v, true
end

function all_modes(f :: KDE, min_v, min_peak_radius = f.K.σ,  newton_radius = f.K.σ, iters = 20)
  points = f.points
  maxx = maximum(abs(p[1]) for p ∈ points)
  maxy = maximum(abs(p[2]) for p ∈ points)
  peak_tree = FixedRadiusCellList(SVector{2, Float64}[], min_peak_radius, maxx, maxy)
  # This is excessive. Can we not skip points near previous points with low function value?
  # Using.. e.g. lipschitz bound on gradient or hessian? or just evaluate on a (fine) grid...? duh?
  for p in points
    if !has_neighbors(peak_tree, p, min_peak_radius)
      (x, _, flag) = _newton(f, p, newton_radius, iters, min_v)
      if flag && !has_neighbors(peak_tree, x, min_peak_radius)
        push!(peak_tree, x)
      end
    end
  end

  # ugly
  r = SVector{2, Float64}[]
  for b in values(peak_tree.cells)
    append!(r, b)
  end
  r
end

function integrate(f :: KDE, (l_x, u_x), (l_y, u_y))
    @assert (u_x - l_x) <= f.K.σ*10 && (u_y - l_y) <= f.K.σ*10
    foldl_range_query(Integrator((l_x, u_x), (l_y, u_y), f.K.σ), 0.0, f.T, SVector((u_x+l_x)/2, (u_y+l_y)/2))
end

struct Integrator
    x_bounds :: NTuple{2, Float64}
    y_bounds :: NTuple{2, Float64}
    σ :: Float64
end

@inline function (s :: Integrator)(n :: SVector{2, Float64}, state :: Float64)
  d_x = Normal(n[1], s.σ)
  d_y = Normal(n[2], s.σ)
  state + (cdf(d_x, s.x_bounds[2]) - cdf(d_x, s.x_bounds[1])) * (cdf(d_y, s.y_bounds[2]) - cdf(d_y, s.y_bounds[1]))
end

render(K :: KDE, x_bins, y_bins) = render_fast(x_bins, y_bins, K.points, K.K.σ, Val(ceil(Int,4*K.K.σ/step(x_bins))))

function render_fast(x_bins, y_bins, points, sigma, ::Val{N}) where {N}
  x_tmp = @MVector zeros(2 * N + 1)
  y_tmp = @MVector zeros(2 * N + 1)

  x_view = @MVector zeros(2 * N)
  y_view = @MVector zeros(2 * N)
  img = zeros(length(x_bins), length(y_bins))
  @inbounds for p in points
    (x, y) = p
    (x_idx, y_idx) = searchsortedfirst(x_bins, x), searchsortedfirst(y_bins, y)
    if N < x_idx < length(x_bins) - N && N < y_idx < length(y_bins) - N
      for (i, x_i) in enumerate(x_idx-N:x_idx+N)
        x_tmp[i] = cdf(Normal(x, sigma), x_bins[x_i+1])
      end

      for (i, x_i) in enumerate(x_idx-N:x_idx+N-1)
        x_view[i] = x_tmp[i+1] - x_tmp[i]
      end

      for (i, y_i) in enumerate(y_idx-N:y_idx+N)
        y_tmp[i] = cdf(Normal(y, sigma), y_bins[y_i+1])
      end

      for (i, y_i) in enumerate(y_idx-N:y_idx+N-1)
        y_view[i] = y_tmp[i+1] - y_tmp[i]
      end

      for (y_v, y_i) in zip(y_view, y_idx-N:y_idx+N-1)
        for (x_v, x_i) in zip(x_view, x_idx-N:x_idx+N-1)
          img[x_i, y_i] += x_v * y_v
        end
      end
    end
  end
  img
end


# struct KDE
#   K :: GaussianKernel
#   T :: FixedRadiusCellList
#   points :: Vector{SVector{2, Float64}}
# end


@inline function _gauss_inner_product(μ_1, μ_2, σ_1, σ_2)
  (1/sqrt((σ_1^2 + σ_2^2)*2*π))*exp(-0.5(μ_1-μ_2)^2/(σ_1^2+ σ_2^2))
end

struct InnerProductAccumulator
    mul :: Float64
    μ :: SVector{2,Float64}
end

@inline function (ip :: InnerProductAccumulator)(x :: SVector{2, Float64}, state :: Float64)
  d = x - ip.μ
  state + exp(ip.mul*(d[1]*d[1]+d[2]*d[2]))
end

function inner_product(K_1 :: KDE, K_2 :: KDE)
  σ_1 = K_1.K.σ
  σ_2 = K_2.K.σ
  if σ_1 > σ_2
    inner_product(K_2, K_1)
  else
    mul = -0.5/(σ_1^2+ σ_2^2)
    r = 0.0
    for p in K_2.points
      r += foldl_range_query(InnerProductAccumulator(mul, p), 0.0, K_1.T, p)
    end
    r/((σ_1^2 + σ_2^2)*2*π)
  end
end


struct GradInnerProductAccumulator
    mul :: Float64
    μ :: SVector{2,Float64}
    o :: SVector{2, Float64}
end

@inline function (ip :: GradInnerProductAccumulator)(x :: SVector{2, Float64}, (value, gradient) :: Tuple{Float64, SVector{2,Float64}})
  d = x + ip.o - ip.μ
  v = exp(ip.mul*(d[1]*d[1]+d[2]*d[2]))
  (value + v, gradient + ip.mul*v*d)
end

function inner_product_offset_gradient(K_1::KDE, K_2, o)
  σ_1 = K_1.K.σ
  σ_2 = K_2.K.σ
  @assert σ_1 ≥ σ_2

  mul = -0.5 / (σ_1^2 + σ_2^2)
  v = 0.0
  g = SVector(0.0, 0.0)
  for p in K_2.points
    (d_v, d_g) = foldl_range_query(
      GradInnerProductAccumulator(mul, p, o),
      (0.0, SVector(0.0, 0.0)),
      K_1.T,
      p
    )
    v += d_v
    g += d_g
  end
  v / ((σ_1^2 + σ_2^2) * 2 * π), 2*g / ((σ_1^2 + σ_2^2) * 2 * π)
end


end
