import numpy as np
def pipe_fem(E, nu, pressure, D_outer, t_wall, damage_factor):
    t_effective = t_wall * damage_factor
    R_inner = D_outer / 2 - t_effective
    hoop_stress = pressure * R_inner / t_effective
    axial_stress = pressure * R_inner / (2 * t_effective)
    radial_stress = -pressure/2

    von_mises_stress = np.sqrt(hoop_stress**2 - hoop_stress*axial_stress + axial_stress**2 + radial_stress**2)
    yield_strength = 414e6
    utilization = von_mises_stress / yield_strength
    return hoop_stress, axial_stress, radial_stress, von_mises_stress, utilization

    