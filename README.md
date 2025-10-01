# UnloadMesh

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A FEniCS-based toolkit for computing the unloaded (zero-stress) configuration of a pre-strained geometry. This project implements an iterative fixed-point algorithm to solve the inverse mechanics problem, essential for accurate biomechanical and soft tissue simulations.

## Motivation

In biomechanics, patient-specific geometries are often obtained from medical imaging techniques like CT or MRI. A critical challenge is that these images capture the organ in its *in vivo* state, meaning it is already under physiological loads (e.g., blood pressure). Performing finite element simulations directly on this loaded geometry leads to inaccurate stress distributions and incorrect deformation analyses.

To achieve physically accurate simulations, it is necessary to solve the inverse problem: finding the "zero-pressure" or "unloaded" configuration of the geometry. This repository provides a practical implementation of the **backward displacement method** proposed by Bols et al. (2013) to solve this problem iteratively.

## Features

* Implementation of the backward displacement fixed-point algorithm.
* Built on the robust **FEniCS Project** (`dolfin`) for finite element modeling.
* Integration with **Guccione's anisotropic material model** for cardiac tissue simulations.
* Support for rule-based fiber orientation generation via the **LDRB** library.

## Getting Started

### Prerequisites

This project requires a working installation of the legacy FEniCS Project (`dolfin`). Please follow the official installation instructions.

Key Python libraries used:
* `dolfin`
* `numpy`
* `ldrb`
* `guccionematerial` (based on the [Simula-SSCP lectures](https://github.com/Simula-SSCP/SSCP_lectures/tree/main))

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/DaniDanyDani/UnloadMesh.git](https://github.com/DaniDanyDani/UnloadMesh.git)
    cd UnloadMesh
    ```
2.  Ensure that all required modules, including your custom solvers (`solver_lv`, `guccionematerial`), are available in your Python path.

## Usage

The main workflow involves loading your *in vivo* mesh and pressure data, and then calling the iterative solver.

```python
import dolfin as df
import numpy as np
import os

# Custom modules for the simulation
from solver_lv import solve_inflation
from guccionematerial import GuccioneMaterial
import ldrb

# 1. Load your in-vivo (loaded) mesh
mesh_invivo = df.Mesh("path/to/your/invivo_mesh.xml")

# 2. Define material properties and in-vivo pressure
# (Example using Guccione material)
material = GuccioneMaterial(...)
invivo_pressure = 80.0  # Example pressure in mmHg

# 3. Set up the fixed-point solver parameters
# ... (solver setup) ...

# 4. Run the algorithm to find the unloaded mesh
mesh_unloaded = find_unloaded_configuration(mesh_invivo, invivo_pressure, material)

# 5. Save the resulting zero-pressure mesh
file = df.File("unloaded_mesh.pvd")
file << mesh_unloaded
