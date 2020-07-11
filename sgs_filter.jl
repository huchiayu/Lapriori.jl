using HDF5
using LinearAlgebra
using StaticArrays
using Statistics
using Random
using .Threads
push!(LOAD_PATH, pwd())
using Octree

const fac_filter = 4
const BOXSIZE_X = 1.
const BOXSIZE_Y = 1.
const BOXSIZE_Z = 1.
const boxsizes = SVector(BOXSIZE_X, BOXSIZE_Y, BOXSIZE_Z)
const center = SVector(BOXSIZE_X, BOXSIZE_Y, BOXSIZE_Z) .* 0.5
const topnode_length = SVector(BOXSIZE_Z, BOXSIZE_Z, BOXSIZE_Z)


const T = Float64
const N = 3
const GEO_FAC = T(4.0 / 3.0 * pi)
const KERNELCONST = T(16.0/pi)
@inline ramp(x) = max(0, x);
@inline function kernel_cubic(x::T) where {T}
    return T(KERNELCONST * (ramp(1.0 - x)^3 - 4.0 * ramp(0.5 - x)^3))
end

function read_snap(filename, T)

    header = h5readattr(filename, "/Header")
    boxsize = header["BoxSize"]
    time    = header["Time"]

    N_gas::Int = header["NumPart_ThisFile"][1]

    pos_gas::Matrix{T} = h5read(filename, "PartType0/Coordinates");
    vel_gas::Matrix{T} = h5read(filename, "PartType0/Velocities");
    rho::Vector{T}     = h5read(filename, "PartType0/Density");
    u::Vector{T}       = h5read(filename, "PartType0/InternalEnergy");
    m_gas::Vector{T}   = h5read(filename, "PartType0/Masses");
    hsml::Vector{T}    = h5read(filename, "PartType0/SmoothingLength");
    scal::Vector{T}    = h5read(filename, "PartType0/PassiveScalarField");

    id_gas::Vector{Int} = h5read(filename, "PartType0/ParticleIDs");

    return header, N_gas, pos_gas, vel_gas, rho, u, m_gas, hsml, scal, id_gas
end

function vec2svec(vec::Matrix{T}) where {T}
    svec = [SVector{3,T}(vec[:,i]) for i in 1:size(vec,2)]
end
function mat2smat(mat::Array{T,3}) where {T}
    smat = [SMatrix{3,3,T}(mat[:,:,i]) for i in 1:1:size(vec,3)]
end

function get_hsml(X::Array{SVector{N,T},1}, tree::Node{N,T}, hsml::Vector{T}) where {N,T}
    N_gas::Int = length(X)
    ngb_flag = zeros(Int, N_gas)
    N_ngbs = zeros(Int, N_gas)

    N_redo::Int = 1
    cc = 0
    @time while N_redo > 0
        N_redo = 0
        @threads for idx in ridx
            if ngb_flag[idx] == 0
                idx_ngbs = get_gather_ngb_tree(X[idx], hsml[idx], tree, boxsizes)
                N_ngbs[idx] = length(idx_ngbs)
                dNgbs = N_ngbs[idx] - N_ngbs_target
                if abs(dNgbs) <= dNgbs_tol
                    ngb_flag[idx] = 1
                    #@show "converged!", idx, length(idx_ngbs), hsml[idx]
                else
                    N_redo += 1
                    fac = dNgbs > 0 ? 0.9 : 1.1
                    hsml[idx] *= fac
                    #@show idx, length(idx_ngbs), hsml[idx]
                end
            end
        end
        cc += 1
        println("iteration ", cc, "... need to redo ", N_redo, " particles...")
    end
    return hsml
end


function get_smoothed_field(field::Vector{T2}, mass::Vector{T}, rho::Vector{T},
                            X::Array{SVector{N,T},1}, hsml::Vector{T}, tree::Node{N,T}, ridx::Vector{Int}) where {N,T,T2}
    N_gas::Int = length(X)
    smoothed_field = zeros(T2, N_gas)  #T2 can be Float, SVector or SMatrix
    volume = mass ./ rho

    @threads for i::Int in ridx
        idx_ngbs = get_gather_ngb_tree(X[i], hsml[i], tree, boxsizes)
        for k in eachindex(idx_ngbs)
            j = idx_ngbs[k]
            dx = nearest.(X[j] - X[i], boxsizes)
            dr = norm(dx)
            Wij = kernel_cubic(dr/hsml[i]) #divide by hsml[i]^N after the loop
            smoothed_field[i] += field[j] * (volume[j] * Wij)
        end
        smoothed_field[i] /= hsml[i]^N
    end
    return smoothed_field
end

function get_Favre_smoothed_field(field::Vector{T2}, mass::Vector{T}, f_rho::Vector{T},
                                X::Array{SVector{N,T},1}, hsml::Vector{T}, tree::Node{N,T}, ridx::Vector{Int}) where {N,T,T2}
    N_gas::Int = length(X)
    smoothed_field = zeros(T2, N_gas)  #T2 can be Float, SVector or SMatrix

    @threads for i::Int in ridx
        idx_ngbs = get_gather_ngb_tree(X[i], hsml[i], tree, boxsizes)
        for k in eachindex(idx_ngbs)
            j = idx_ngbs[k]
            dx = nearest.(X[j] - X[i], boxsizes)
            Wij = kernel_cubic(norm(dx)/hsml[i]) #divide by hsml[i]^N after the loop
            smoothed_field[i] += field[j] * (mass[j] * Wij)
        end
        smoothed_field[i] /= (hsml[i]^N * f_rho[i])
    end
    return smoothed_field
end


function calc_gradient(field::Vector{T}, X::Array{SVector{N,T},1}, hsml::Vector{T}, tree::Node{N,T}) where {N,T}
    N_gas::Int = length(X)
    grad_field = zeros(SVector{N,T}, N_gas)

    @threads for i in 1:N_gas
        idx_ngbs = get_gather_ngb_tree(X[i], hsml[i], tree, boxsizes)
        @assert length(idx_ngbs) > 0

        renorm_matrix = @SMatrix zeros(T,N,N)
        bvec = @SVector zeros(T,N)

        for k in eachindex(idx_ngbs)
            j = idx_ngbs[k]
            dx = nearest.(X[j] - X[i], boxsizes)
            dr = norm(dx)
            W = kernel_cubic(dr/hsml[i]) #don't need to divide by hsml[i]^N as this will cancel out
            renorm_matrix += (dx * dx') .* W
            df = field[j] - field[i]
            bvec +=  (df * W) .* dx
        end
        renorm_matrix = inv(renorm_matrix)
        grad_field[i] = renorm_matrix * bvec
    end
    return grad_field
end


function calc_filtered_fields(fname_in::String, fname_out::String)

    @time header, N_gas, pos, vel, rho, u, mass, hsml, scal, id_gas = read_snap(fname_in, T);

    X = vec2svec(pos);
    Vel = vec2svec(vel);
    VelVel = Vel .* transpose.(Vel);


    #for each particle, we still use all particles within the kernel to get the filtered fields
    #however, we only need to do this for N_gas/Ns particles
    Ns = fac_filter^3  # downsampling factor
    rng = MersenneTwister(1114);
    ridx = randperm(rng, N_gas)[1:div(N_gas,Ns)];

    @time tree = buildtree(X, hsml, mass, mass, mass, center, topnode_length);
    f_hsml = fac_filter .* hsml;

    VelVel = Vel.*transpose.(Vel)
    ScalVel = scal.*Vel

    @time f_rho = get_smoothed_field(rho, mass, rho, X, f_hsml, tree, ridx)
    @time fa_scal    = get_Favre_smoothed_field(scal   , mass, f_rho, X, f_hsml, tree, ridx)
    @time fa_Vel     = get_Favre_smoothed_field(Vel    , mass, f_rho, X, f_hsml, tree, ridx)
    @time fa_VelVel  = get_Favre_smoothed_field(VelVel , mass, f_rho, X, f_hsml, tree, ridx)
    @time fa_ScalVel = get_Favre_smoothed_field(ScalVel, mass, f_rho, X, f_hsml, tree, ridx)

    ### down-sample ###
    println("Down-sample begins... N_gas = ", N_gas)
    f_hsml = f_hsml[ridx]
    f_rho = f_rho[ridx]
    fa_Vel = fa_Vel[ridx]
    fa_scal = fa_scal[ridx]
    fa_VelVel = fa_VelVel[ridx]
    fa_ScalVel = fa_ScalVel[ridx]
    X = X[ridx]
    N_gas = div(N_gas, Ns)
    println("Down-sample finished!!! N_gas = ", N_gas)
    dummy=ones(N_gas)

    #rebuild a low-res tree after downsampling
    tree_ds = buildtree(X, f_hsml, dummy, dummy, dummy, center, topnode_length);

    @time grad_fa_scal = calc_gradient(fa_scal, X, f_hsml, tree_ds);
    @time grad_fa_vx = calc_gradient(getindex.(fa_Vel,1), X, f_hsml, tree_ds);
    @time grad_fa_vy = calc_gradient(getindex.(fa_Vel,2), X, f_hsml, tree_ds);
    @time grad_fa_vz = calc_gradient(getindex.(fa_Vel,3), X, f_hsml, tree_ds);

    fid=h5open(fname_out,"w")
    grp_part = g_create(fid,"SGS");
    h5write(fname_out, "SGS/FilteredSmoothingLength" , f_hsml)
    h5write(fname_out, "SGS/FilteredDensity"         , f_rho)
    h5write(fname_out, "SGS/FilteredScalar"          , fa_scal)
    h5write(fname_out, "SGS/FilteredVelocity"        , reshape(reinterpret(T,fa_Vel),3,N_gas))
    h5write(fname_out, "SGS/FilteredVelVel"          , reshape(reinterpret(T,fa_VelVel),3,3,N_gas))
    h5write(fname_out, "SGS/FilteredVelocityScalar"  , reshape(reinterpret(T,fa_ScalVel),3,N_gas))
    h5write(fname_out, "SGS/FilteredScalarGradient"  , reshape(reinterpret(T,grad_fa_scal),3,N_gas))
    h5write(fname_out, "SGS/FilteredVxGradient"      , reshape(reinterpret(T,grad_fa_vx),3,N_gas))
    h5write(fname_out, "SGS/FilteredVyGradient"      , reshape(reinterpret(T,grad_fa_vy),3,N_gas))
    h5write(fname_out, "SGS/FilteredVzGradient"      , reshape(reinterpret(T,grad_fa_vz),3,N_gas))
    close(fid)

end
