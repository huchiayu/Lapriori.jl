include("sgs_filter.jl")
include("sgs_apriori.jl")

#snaps = 200:5:600
file_path = "/Users/chu/simulations/turbbox/N128/"
snaps = 701:701

for k in snaps
    println(k)
    if k < 10
        num = "00" * string(k)
    elseif k < 100
        num = "0" * string(k)
    else
        num = string(k)
    end
    fname_in = file_path * "/snap_"*num*".hdf5"
    fname_out = file_path * "/sgs_fields_" * num * ".hdf5"
    @time calc_filtered_fields(fname_in, fname_out)

    fname_in = fname_out
    fname_out = file_path * "/sgs_model_" * num * ".hdf5"
    @time q_true, q_eddy, q_grad, tau_true, tau_eddy, tau_grad = calc_sgs_fluxes(fname_in, fname_out)
end
