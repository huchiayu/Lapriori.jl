# Lapriori.jl
performs "a priori" analysis of SGS turbulence models for Lagrangian hydro codes

## Filtered fields and gradients
The function
```
calc_filtered_fields(fname_in, fname_out)
```
calculates filtered fields using a cubic spline kernel. 
The gradients are calculated using the least-square technique which is exact for linear functions.

* ```fname_in```: input file, a Gadget snapshot of format 3 (HDF5).

* ```fname_out```: output file of the filtered fields in HDF5 format.


## SGS fluxes (true vs. modeled)
The function
```
calc_sgs_fluxes(fname_in, fname_out)
```
calculates the true and modeled SGS fluxes given the filtered SGS fields.

* ```fname_in```: the filtered fields obtained from ```calc_sgs_fields```.

* ```fname_out```: the true and modeled SGS fluxes in HDF5 format.
