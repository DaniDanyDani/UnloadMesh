import dolfin as df
import numpy as np
import os
import matplotlib.pyplot as plt
from src.solver import solve_inflation_lvrv
from src.utils import compute_cavity_volume, plot_convergence_history

TOLERANCIA = 1e-3      
MAX_ITERACOES = 50     
PRESSAO_MEDIDA = [1.8,0.8]  
SOLVER_PRESSURE_STEPS = 50

# --- Inputs
MESH_PATH = "./data/Patient_2_ed/Patient_2_ed.xml"
FFUN_PATH = "./data/Patient_2_ed/Patient_2_ed_facet_region.xml"
OUTPUT_DIR = "results_unload/Patient_2_ed"
UNLOADED_MESH_FILE = os.path.join(OUTPUT_DIR, "unloaded_mesh.xdmf")
ITERATIVE_DISP_FILE = os.path.join(OUTPUT_DIR, "deslocamento_iterativo.pvd")
CONVERGENCE_GRAPH = os.path.join(OUTPUT_DIR, "convergence_graph.png")

# --- Markers para ldrb
ldrb_markers = {"base": 10, "lv": 20, "epi": 40, "rv": 30}

print("Inicializando o processo para encontrar a geometria sem carga...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Carregando malha inicial
try:
    mesh = df.Mesh(MESH_PATH)
    ffun = df.MeshFunction("size_t", mesh, FFUN_PATH)
except RuntimeError:
    print(f"\nERRO: Malha '{MESH_PATH}' ou fronteiras '{FFUN_PATH}' não encontradas.")
    exit()

# --- Iniciando salvamento iterativo das malhas
disp_file = df.File(ITERATIVE_DISP_FILE)


# --- iniciando malhas iterativas
xm = mesh.coordinates()[:].copy()
X = xm.copy()
x = xm.copy()

fiber, sheet, sheet_normal = None, None, None


# --- Iniciando método do ponto fixo
volumes_por_iteracao = []
residuos_por_iteracao = []
for i in range(MAX_ITERACOES):
    print(f"\n--- Iteração {i+1}/{MAX_ITERACOES} ---")

    try:
        # --- Atualizar a malha antes de iniciar simulação
        mesh.coordinates()[:] = X.copy()
        mesh.bounding_box_tree().build(mesh)
        # cells_colision = mesh.compute_entity_collisions(mesh)

        # --- Iniciando simulação, chamando solver
        print("Executando simulação direta (chamando o solver)...")
        u_calculado, [fiber, sheet, sheet_normal] = solve_inflation_lvrv(
            mesh, ffun, ldrb_markers, PRESSAO_MEDIDA, SOLVER_PRESSURE_STEPS 
        )

        # --- renomeando o campo u para o salvamento iterativo
        u_calculado.rename("u", f"displacement_iter_{i+1}")
        disp_file << u_calculado
        
        # --- transformando o campo de deslocamento em um array
        coords = mesh.coordinates()
        u_array = np.array([u_calculado(xx) for xx in coords])

        # --- Adquirindo malha expandida
        x = mesh.coordinates()[:].copy()
        x += u_array

        # --- Cálculo do resíduo e salvando em uma lista para plotar depois
        lres = []
        for i in range(np.shape(xm)[0]):
            lres.append(np.linalg.norm(xm[i,:]-x[i,:], 2))
        res = max(lres)
        residuos_por_iteracao.append(res)

        # --- Calculando volume e salvando em uma lista para plotar depois
        volume_calculado_lv = compute_cavity_volume(mesh, ffun, ldrb_markers, "lv", u_calculado)
        volume_calculado_rv = compute_cavity_volume(mesh, ffun, ldrb_markers, "rv", u_calculado)
        volumes_por_iteracao.append([volume_calculado_lv, volume_calculado_rv])
        
        print(f"Resíduo máximo nesta iteração: {res:.6f}")

        if res < TOLERANCIA:
            print("\nConvergência atingida com sucesso!")
            break

        # --- Atualizando geometria para a próxima iteração
        print("Atualizando a estimativa da geometria descarregada com sub-relaxação...")
        X = xm.copy() - (u_array)

    except RuntimeError as e:
        print(f"\nERRO na simulação. O solver não convergiu na iteração {i+1}.")
        print("Detalhes do erro do FEniCS:", e)
        break

else: 
    print(f"\nAVISO: Número máximo de {MAX_ITERACOES} iterações atingido sem convergência total.")

print(f"\nSalvando a geometria descarregada final em '{UNLOADED_MESH_FILE}'...")
with df.XDMFFile(UNLOADED_MESH_FILE) as outfile:
    outfile.write(mesh)
    u_calculado.rename("displacement", "displacement")
    fiber.rename("f", "f")
    sheet.rename("s", "s")
    sheet_normal.rename("n", "n")
    outfile.write(fiber, 0)
    outfile.write(sheet, 0)
    outfile.write(sheet_normal, 0)
    outfile.write(u_calculado, 0)
print("Processo concluído.")

# --- VISUALIZAÇÃO DOS RESULTADOS ---
if volumes_por_iteracao and residuos_por_iteracao:
    plot_convergence_history(volumes_por_iteracao, residuos_por_iteracao, CONVERGENCE_GRAPH)
