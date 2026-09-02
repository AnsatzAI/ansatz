# Two-transmon unit cell: two qubit+readout pairs hanging off one shared
# feedline — the minimal industrially-shaped frequency-planning geometry.
# Adapted from DeviceLayout.jl's SingleTransmon example (examples/SingleTransmon).
#
# Usage (cwd = palace/examples/transmon):
#   julia --project gen_two_transmon.jl probe            # render + print groups
#   julia --project gen_two_transmon.jl spec.json        # mesh + Palace config
#
# Spec: {"tag": "t001", "solver_order": 2, "amr_iterations": 0,
#        "params_um": {"cap_length_1": 620, "cap_length_2": 560,
#                       "total_length_1": 5000, "total_length_2": 4600,
#                       "l_claw_1": 121, "l_claw_2": 121,
#                       "cap_gap_1": 30, "cap_gap_2": 30},
#        "params_int": {"n_meander_turns": 5}}

module TwoTransmon

using FileIO, JSON
using DeviceLayout, DeviceLayout.SchematicDrivenLayout, DeviceLayout.PreferredUnits
import .SchematicDrivenLayout.ExamplePDK
import .SchematicDrivenLayout.ExamplePDK: LayerVocabulary, add_bridges!
using .ExamplePDK.Transmons, .ExamplePDK.ReadoutResonators
import .ExamplePDK.SimpleJunctions: ExampleSimpleJunction

function two_transmon(;
    cap_length_1=620μm, cap_length_2=560μm,
    cap_gap_1=30μm, cap_gap_2=30μm,
    total_length_1=5000μm, total_length_2=4600μm,
    l_claw_1=121μm, l_claw_2=121μm,
    n_meander_turns=5,
    hanger_length=500μm, bend_radius=50μm,
    w_shield=2μm, claw_gap=6μm, w_claw=34μm, cap_width=24μm,
    save_mesh=false, mesh_path="two_transmon.msh2",
)
    reset_uniquename!()
    cpw_width = 10μm
    cpw_gap = 6μm
    PATH_STYLE = Paths.SimpleCPW(cpw_width, cpw_gap)
    BRIDGE_STYLE = ExamplePDK.bridge_geometry(PATH_STYLE)
    coupling_gap = 5μm
    arm_length = 428μm

    function mk_pair(idx, cap_length, cap_gap, total_length, l_claw)
        w_grasp = cap_width + 2 * cap_gap
        total_height =
            arm_length + coupling_gap + Paths.extent(PATH_STYLE) +
            hanger_length + (3 + n_meander_turns * 2) * bend_radius
        qubit = ExampleRectangleTransmon(;
            jj_template=ExampleSimpleJunction(),
            name="qubit$(idx)", cap_length, cap_gap, cap_width)
        rres = ExampleClawedMeanderReadout(;
            name="rres$(idx)",
            coupling_length=400μm, coupling_gap, total_length,
            w_shield, w_claw, l_claw, claw_gap, w_grasp,
            n_meander_turns, total_height, hanger_length, bend_radius,
            bridge=BRIDGE_STYLE)
        return qubit, rres
    end

    q1, r1 = mk_pair(1, cap_length_1, cap_gap_1, total_length_1, l_claw_1)
    q2, r2 = mk_pair(2, cap_length_2, cap_gap_2, total_length_2, l_claw_2)

    readout_length = 3600μm
    p_readout = Path(
        Point(0μm, 0μm);
        α0=π / 2, name="p_ro", metadata=LayerVocabulary.METAL_NEGATIVE)
    straight!(p_readout, readout_length / 2, PATH_STYLE)
    straight!(p_readout, readout_length / 2, PATH_STYLE)

    csport = CoordinateSystem(uniquename("port"), nm)
    render!(csport,
        only_simulated(centered(Rectangle(cpw_width, cpw_width))),
        LayerVocabulary.PORT)
    attach!(p_readout, sref(csport), cpw_width, i=1)
    attach!(p_readout, sref(csport), readout_length / 2 - cpw_width, i=2)

    g = SchematicGraph("two-transmon")
    q1n = add_node!(g, q1)
    r1n = fuse!(g, q1n, r1)
    q2n = add_node!(g, q2)
    r2n = fuse!(g, q2n, r2)
    pn = add_node!(g, p_readout)
    # hangers at 1/4 and 3/4 of the feedline, same side
    attach!(g, pn, r1n => :feedline, readout_length / 4, i=1, location=1)
    attach!(g, pn, r2n => :feedline, readout_length / 4, i=2, location=1)

    floorplan = plan(g)
    add_bridges!(floorplan, BRIDGE_STYLE, spacing=300μm)

    substrate_x = 4mm
    substrate_y = 4.8mm
    center_xyz = DeviceLayout.center(floorplan)
    chip = centered(Rectangle(substrate_x, substrate_y), on_pt=center_xyz)
    sim_area = centered(Rectangle(substrate_x, substrate_y), on_pt=center_xyz)
    render!(floorplan.coordinate_system, sim_area, LayerVocabulary.SIMULATED_AREA)
    render!(floorplan.coordinate_system, sim_area, LayerVocabulary.WRITEABLE_AREA)
    render!(floorplan.coordinate_system, chip, LayerVocabulary.CHIP_AREA)
    check!(floorplan)

    groups = ["port_1", "port_2", "lumped_element", "lumped_element_1",
              "lumped_element_2"]
    tech = ExamplePDK.singlechip_solidmodel_target(groups...)
    sm = SolidModel("two_transmon", overwrite=true)
    SolidModels.set_gmsh_option("General.Verbosity", 1)
    SolidModels.mesh_order(2)
    render!(sm, floorplan, tech)

    if save_mesh
        SolidModels.gmsh.model.mesh.generate(3)
        save(mesh_path, sm)
    end
    return sm
end

function configfile2(sm; solver_order=2, amr=0, mesh_rel="mesh/two.msh2",
                     out_rel="postpro/two", lj_nh=14.860)
    attributes = SolidModels.attributes(sm)
    jj_keys = sort([k for k in keys(attributes)
                    if occursin(r"^lumped_element_\d+$", k)])
    isempty(jj_keys) && (jj_keys = ["lumped_element"])
    ports = Any[]
    for (i, pk) in enumerate(["port_1", "port_2"])
        push!(ports, Dict("Index" => i, "Attributes" => [attributes[pk]],
                          "R" => 50, "Direction" => "+X"))
    end
    for (j, jk) in enumerate(jj_keys)
        push!(ports, Dict("Index" => 2 + j, "Attributes" => [attributes[jk]],
                          "L" => lj_nh * 1e-9, "C" => 5.5e-15,
                          "Direction" => "+Y"))
    end
    return Dict(
        "Problem" => Dict("Type" => "Eigenmode", "Verbose" => 2,
                          "Output" => out_rel),
        "Model" => Dict("Mesh" => mesh_rel, "L0" => 1e-6,
                        "Refinement" => Dict("MaxIts" => amr)),
        "Domains" => Dict(
            "Materials" => [
                Dict("Attributes" => [attributes["vacuum"]],
                     "Permeability" => 1.0, "Permittivity" => 1.0),
                Dict("Attributes" => [attributes["substrate"]],
                     "Permeability" => [0.99999975, 0.99999975, 0.99999979],
                     "Permittivity" => [9.3, 9.3, 11.5],
                     "LossTan" => [3.0e-5, 3.0e-5, 8.6e-5],
                     "MaterialAxes" => [[0.8, 0.6, 0.0], [-0.6, 0.8, 0.0],
                                        [0.0, 0.0, 1.0]])],
            "Postprocessing" => Dict(
                "Energy" => [Dict("Index" => 1,
                                  "Attributes" => [attributes["substrate"]])])),
        "Boundaries" => Dict(
            "PEC" => Dict("Attributes" => [attributes["metal"]]),
            "Absorbing" => Dict(
                "Attributes" => [attributes["exterior_boundary"]],
                "Order" => 1),
            "LumpedPort" => ports),
        "Solver" => Dict(
            "Order" => solver_order,
            "Eigenmode" => Dict("N" => 8, "Tol" => 1.0e-8, "Target" => 3.2,
                                "Save" => 0, "MaxSize" => 80),
            "Linear" => Dict("Type" => "Default", "Tol" => 1.0e-12,
                             "MaxIts" => 600)))
end

end # module

using .TwoTransmon
import JSON
using DeviceLayout: μm
import DeviceLayout.SolidModels as SolidModels

function main(arg::AbstractString)
    if arg == "probe"
        sm = TwoTransmon.two_transmon(save_mesh=false)
        println("GROUPS: ", sort(collect(keys(SolidModels.attributes(sm)))))
        return
    end
    spec = JSON.parsefile(arg)
    tag = spec["tag"]
    kwargs = Dict{Symbol,Any}()
    for (k, v) in get(spec, "params_um", Dict())
        kwargs[Symbol(k)] = Float64(v) * μm
    end
    for (k, v) in get(spec, "params_int", Dict())
        kwargs[Symbol(k)] = Int(v)
    end
    mesh_path = joinpath(pwd(), "mesh", "$(tag).msh2")
    sm = TwoTransmon.two_transmon(; save_mesh=true, mesh_path, kwargs...)
    cfg = TwoTransmon.configfile2(sm;
        solver_order=get(spec, "solver_order", 2),
        amr=get(spec, "amr_iterations", 0),
        mesh_rel=joinpath("mesh", "$(tag).msh2"),
        out_rel=joinpath("postpro", "ansatz", tag))
    open(joinpath(pwd(), "ansatz_$(tag).json"), "w") do f
        JSON.print(f, cfg, 1)
    end
    println("GENERATED $(tag)")
end

main(ARGS[1])
