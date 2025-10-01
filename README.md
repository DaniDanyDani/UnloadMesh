UnloadMeshA FEniCS-based implementation of the fixed-point iteration method to determine the unloaded (zero-stress) configuration of pre-strained biological structures, such as blood vessels and cardiac tissue.MotivationIn biomechanics, patient-specific geometries are often obtained from medical imaging techniques like CT or MRI. A critical challenge is that these images capture the organ in its in vivo state, meaning it is already under physiological loads (e.g., blood pressure). Performing finite element simulations directly on this loaded geometry leads to inaccurate stress distributions and incorrect deformation analyses.To achieve physically accurate simulations, it is necessary to solve the inverse problem: finding the "zero-pressure" or "unloaded" configuration of the geometry. This repository provides a practical implementation of the backward displacement method proposed by Bols et al. (2013) to solve this problem iteratively.MethodologyThe core of this project is a fixed-point iteration algorithm to find the zero-pressure geometry () from a known in vivo (loaded) geometry () and a known internal pressure ().The algorithm, as described by Bols et al., can be summarized as follows:Initialization: The first guess for the unloaded geometry () is the loaded geometry itself: .Iteration Loop: For each iteration i:a. Forward Simulation: A standard finite element analysis is performed. The current guess for the unloaded geometry, Ω(Xi,0), is subjected to the known in vivo pressure load, pm​. This computes a new deformed configuration, Ω(xi,σi).b. Calculate Displacements: The displacement field Ui is calculated as the difference between the resulting deformed geometry and the current guess: Ui=xi−Xi.c. Update Guess: A new, improved guess for the unloaded geometry (Xi+1) is calculated by subtracting the displacement field (Ui) from the original in vivo geometry (xm​): Xi+1=xm​−Ui.Convergence: The loop continues until the distance between the simulated geometry () and the target in vivo geometry () is smaller than a defined tolerance .Upon convergence,  is the desired zero-pressure geometry (), and  is the corresponding in vivo stress tensor field ().FeaturesImplementation of the backward displacement fixed-point algorithm.Built on the robust FEniCS Project (dolfin) for finite element modeling.Integration with Guccione's anisotropic material model for cardiac tissue simulations.Support for rule-based fiber orientation generation via the LDRB library.Getting StartedPrerequisitesThis project requires a working installation of the legacy FEniCS Project (dolfin). Please follow the official installation instructions.Key Python libraries used:dolfinnumpyldrbguccionematerial (based on the Simula-SSCP lectures)InstallationClone the repository:git clone [https://github.com/seu-usuario/UnloadMesh.git](https://github.com/seu-usuario/UnloadMesh.git)
cd UnloadMesh
Ensure that all required modules, including your custom solvers (solver_lv, guccionematerial), are available in your Python path.UsageThe main workflow involves loading your in vivo mesh and pressure data, and then calling the iterative solver.import dolfin as df
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
CitationThis work is based on the methods described in the following paper. If you use this code in your research, please cite the original publication:J. Bols, J. Degroote, B. Trachet, B. Verhegghe, P. Segers, J. Vierendeels, A computational method to assess the in vivo stresses and unloaded configuration of patient-specific blood vessels, Journal of Computational and Applied Mathematics, Volume 246, 2013, Pages 10-17, ISSN 0377-0427, doi:10.1016/j.cam.2012.10.034.@article{BOLS201310,
  title = {A computational method to assess the in vivo stresses and unloaded configuration of patient-specific blood vessels},
  journal = {Journal of Computational and Applied Mathematics},
  volume = {246},
  pages = {10-17},
  year = {2013},
  author = {J. Bols and J. Degroote and B. Trachet and B. Verhegghe and P. Segers and J. Vierendeels}
}
Please also consider citing this repository if it was useful for your work.LicenseThis project is licensed under the MIT License - see the LICENSE file for details.