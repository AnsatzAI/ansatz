"""ansatz: learned solver routing for superconducting qubit design.

Core pipeline:
    geometry  -> planar layouts anchored to experimentally validated design ranges
    pde       -> electrostatic discretization, capacitance extraction, lumped-oscillator model
    solvers   -> classical iterative/direct solvers exposed as composable operators
    surrogate -> neural field predictors used as initializers and mid-solve correctors
    router    -> learned policy choosing the cheapest path to a verified tolerance
    bench     -> time-to-tolerance benchmarking against practitioner baselines
"""

__version__ = "0.1.0"
