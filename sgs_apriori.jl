using HDF5
using PyPlot
using LinearAlgebra
using StaticArrays
using Random

const T = Float64

@inline function get_matrix_norm(A::Matrix{T}) where {T}
    return abs(tr(A * transpose(A)))
end

@inline function gradV_decompose(gradV::SMatrix{T}) where {T}
    divV = tr(gradV)
    shear = 0.5 * (gradV .+ transpose(gradV)) - divV * I / 3.0
    vort = 0.5 * (gradV .- transpose(gradV))
    return divV, shear, vort
end

@inline function outer_product(v1::Vector{T}, v2::Vector{T}) where {T}
    return v1 * v2'
end

function smatrices(M::Array{T,3}) where {T}
    A = reinterpret(SMatrix{3,3,T,9}, reshape(M, (9,size(M,3))))
    return A
end

function calc_sgs_fluxes(fname_in::String, fname_out::String)

    f_rho::Array{T,1} = h5read(fname_in, "SGS/FilteredDensity");
    N_gas::Int32 = Nloop::Int32 = size(f_rho,1)

    f_scal::Array{T,1}     = h5read(fname_in, "SGS/FilteredScalar");
    f_gradScal::Array{T,2} = h5read(fname_in, "SGS/FilteredScalarGradient");
    f_hsml::Array{T,1}     = h5read(fname_in, "SGS/FilteredSmoothingLength");
    f_velvel::Array{T,3}   = h5read(fname_in, "SGS/FilteredVelVel");
    f_vel::Array{T,2}      = h5read(fname_in, "SGS/FilteredVelocity");
    f_velscal::Array{T,2}  = h5read(fname_in, "SGS/FilteredVelocityScalar");
    f_gradVx::Array{T,2}   = h5read(fname_in, "SGS/FilteredVxGradient");
    f_gradVy::Array{T,2}   = h5read(fname_in, "SGS/FilteredVyGradient");
    f_gradVz::Array{T,2}   = h5read(fname_in, "SGS/FilteredVzGradient");

    f_gradV = Array{T}(undef, 3, 3, N_gas);
    f_gradV[1, :, :] = f_gradVx;
    f_gradV[2, :, :] = f_gradVy;
    f_gradV[3, :, :] = f_gradVz;

    tau_true   = Array{T}(undef, 3, 3, N_gas)
    tau_eddy   = Array{T}(undef, 3, 3, N_gas)
    tau_grad   = Array{T}(undef, 3, 3, N_gas)
    shear      = Array{T}(undef, 3, 3, N_gas)
    vort       = Array{T}(undef, 3, 3, N_gas)
    q_grad     = Array{T}(undef, 3, N_gas)
    q_eddy     = Array{T}(undef, 3, N_gas)
    q_true     = Array{T}(undef, 3, N_gas)
    divv       = Array{T}(undef, N_gas)
    shear_norm = Array{T}(undef, N_gas)

    println("start calculation...")

    @inbounds for n in 1:N_gas
        divv[n] = f_gradV[1,1,n] + f_gradV[2,2,n] + f_gradV[3,3,n]
        for i in 1:3, j in 1:3
            vort[j,i,n]  = 0.5 * (f_gradV[j,i,n] - f_gradV[i,j,n])
            shear[j,i,n] = 0.5 * (f_gradV[j,i,n] + f_gradV[i,j,n])
            if(j==i)
                shear[j,i,n] -= divv[n] / 3.0
            end
        end
    end

    s_f_gradV = smatrices(f_gradV)
    shear_s = similar(s_f_gradV)
    vort_s = similar(s_f_gradV)
    divv_s = Vector{T}(undef, N_gas)
    @inbounds for n in 1:N_gas
        divv_s[n], shear_s[n], vort_s[n] = gradV_decompose(s_f_gradV[n])
    end

    @inbounds for n in 1:N_gas
        shear_norm[n] = 0.0
        for i in 1:3, j in 1:3
            shear_norm[n] += shear[j,i,n]^2
        end
        shear_norm[n] = sqrt(shear_norm[n])  #don't forget the sqrt!!!
    end

    @inbounds for n in 1:N_gas
        for i in 1:3
            for j in 1:3
                tau_true[j,i,n] = f_velvel[j,i,n] - (f_vel[j,n] * f_vel[i,n])
            end
        end
    end

    @inbounds for n in 1:N_gas
        for i in 1:3
            for j in 1:3
                tau_grad[j,i,n] = 0.0
                for k in 1:3
                    tau_grad[j,i,n] += (f_gradV[j,k,n] * f_gradV[i,k,n])
                end
                tau_grad[j,i,n] *= f_hsml[n]^2
            end
        end
    end

    @inbounds for n in 1:N_gas
        for i in 1:3
            for j in 1:3
                tau_eddy[j,i,n] = -shear_norm[n] * shear[j,i,n] * f_hsml[n]^2
            end
        end
    end

    @inbounds for n in 1:N_gas
        for i in 1:3
            q_true[i,n] = f_velscal[i,n] - f_scal[n] * f_vel[i,n]
        end
    end

    @inbounds for n in 1:N_gas
        for i in 1:3
            q_grad[i,n] = 0.0
            for j in 1:3
                q_grad[i,n] += (f_gradV[i,j,n] * f_gradScal[j,n])
            end
            q_grad[i,n] *= f_hsml[n]^2
        end
    end

    @inbounds for n in 1:N_gas
        for i in 1:3
            q_eddy[i,n] = -shear_norm[n] * f_gradScal[i,n] * f_hsml[n]^2
        end
    end

    println("done!")

    println("saving data...")
    fid=h5open(fname_out,"w")
    grp_part = g_create(fid,"SGS");
    h5write(fname_out, "SGS/tau_true" , tau_true)
    h5write(fname_out, "SGS/tau_grad" , tau_grad)
    h5write(fname_out, "SGS/tau_eddy" , tau_eddy)
    h5write(fname_out, "SGS/q_true"   , q_true)
    h5write(fname_out, "SGS/q_grad"   , q_grad)
    h5write(fname_out, "SGS/q_eddy"   , q_eddy)
    close(fid)

    println("done")

    return q_true, q_eddy, q_grad, tau_true, tau_eddy, tau_grad

end
