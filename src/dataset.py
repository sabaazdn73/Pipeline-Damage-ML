import numpy as np
import pandas as pd 
from fem_model import pipe_fem 

def generate_dataset(num_samples=1000): 
    data = []
    for _ in range(num_samples): 
        E = np.clip(np.random.normal(200e9, 10e9), 180e9, 220e9)  # Young's modulus
        nu = np.clip(np.random.normal(0.3, 0.01), 0.2, 0.4)   # Poisson's ratio

        pressure = np.random.uniform(5e6, 15e6)  # Pressure in Pascals
        D_outer = np.random.uniform(0.3, 0.9) # Outer diameter in meters
        t_wall = np.random.uniform(0.01, 0.03)  # Wall thickness in meters
        damage_factor = np.random.uniform(0.5, 1.0) # Damage factor (0.5 to 1.0)

        hoop_stress, axial_stress, radial_stress, von_mises_stress, utilization = pipe_fem(E, nu, pressure, D_outer, t_wall, damage_factor)
        
        data.append({
            'E': E,
            'nu': nu,
            'pressure': pressure,
            'D_outer': D_outer,
            't_wall': t_wall,
            'damage_factor': damage_factor,
            'hoop_stress': hoop_stress,
            'axial_stress': axial_stress,
            'radial_stress': radial_stress,
            'von_mises_stress': von_mises_stress,
            'utilization': utilization
        })
    
    df = pd.DataFrame(data)
    return df

def save_dataset(df, path='../data/pipeline_data.csv'): 
    df.to_csv(path, index=False) 
    print(f"Dataset saved to {path}")

if __name__ == "__main__":
    print("Generating dataset...")
    df = generate_dataset(num_samples=1000)
    print(df.head())
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset statistics:")
    print(df.describe())
    save_dataset(df)


