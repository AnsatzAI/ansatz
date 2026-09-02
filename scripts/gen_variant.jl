# Generate one parameterized transmon mesh + Palace config from a JSON spec.
#
# Usage (from palace/examples/transmon):
#   julia --project gen_variant.jl spec.json
#
# Spec: {"tag": "v001", "solver_order": 2,
#        "params_um": {"cap_length": 640.0, "cap_gap": 30.0, ...},
#        "params_int": {"n_meander_turns": 5}}
# Unit-suffixed groups are converted (um -> DeviceLayout micron units).

import JSON

include(joinpath(pwd(), "transmon.jl"))  # run with cwd = palace/examples/transmon
using DeviceLayout: μm

function main(spec_path::AbstractString)
    spec = JSON.parsefile(spec_path)
    tag = spec["tag"]
    kwargs = Dict{Symbol,Any}()
    for (k, v) in get(spec, "params_um", Dict())
        kwargs[Symbol(k)] = Float64(v) * μm
    end
    for (k, v) in get(spec, "params_int", Dict())
        kwargs[Symbol(k)] = Int(v)
    end
    generate_transmon(;
        mesh_filename = "$(tag).msh2",
        config_filename = "ansatz_$(tag).json",
        solver_order = get(spec, "solver_order", 2),
        amr_iterations = get(spec, "amr_iterations", 0),
        kwargs...,
    )
    println("GENERATED $(tag)")
end

main(ARGS[1])
