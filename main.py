# -*- coding: utf-8 -*-

import dolfin as df
import numpy as np
import os
import matplotlib.pyplot as plt
from src.solver import solve_inflation_lv

# --- FUNÇÃO PARA CÁLCULO DE VOLUME ---
def compute_cavity_volume(mesh, mf, numbering, u=None):
    """
    Calcula o volume da cavidade usando uma integral de superfície no endocárdio.
    """
    X = df.SpatialCoordinate(mesh) 
    N = df.FacetNormal(mesh)

    if u is not None:
        I = df.Identity(3)
        F = I + df.grad(u)
        J = df.det(F)
        vol_form = (-1.0/3.0) * df.dot(X + u, J * df.inv(F).T * N)
    else:
        vol_form = (-1.0/3.0) * df.dot(X, N)

    ds = df.Measure('ds', domain=mesh, subdomain_data=mf)
    return df.assemble(vol_form*ds(numbering["lv"]))

TOLERANCIA = 1e-5      
MAX_ITERACOES = 100000     
PRESSAO_MEDIDA = 10.0  
SOLVER_PRESSURE_STEPS = 500


# FATOR_RELAXACAO = 0.2


MESH_PATH = "./data/example/Patient_lv.xml"
FFUN_PATH = "./data/example/Patient_lv_facet_region.xml"
OUTPUT_DIR = "results_unload/teste_8"
UNLOADED_MESH_FILE = os.path.join(OUTPUT_DIR, "unloaded_mesh.xdmf")
ITERATIVE_DISP_FILE = os.path.join(OUTPUT_DIR, "deslocamento_iterativo.pvd")


ldrb_markers = {"base": 10, "lv": 20, "epi": 40, "rv": 30}



print("Inicializando o processo para encontrar a geometria sem carga...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    mesh = df.Mesh(MESH_PATH)
    ffun = df.MeshFunction("size_t", mesh, FFUN_PATH)
    coords_medida = np.copy(mesh.coordinates()) 
except RuntimeError:
    print(f"\nERRO: Malha '{MESH_PATH}' ou fronteiras '{FFUN_PATH}' não encontradas.")
    exit()

disp_file = df.File(ITERATIVE_DISP_FILE)

xm = mesh.coordinates()[:].copy()
X = xm.copy()
x = xm.copy()

volumes_por_iteracao = []
residuos_por_iteracao = []
for i in range(MAX_ITERACOES):
    print(f"\n--- Iteração {i+1}/{MAX_ITERACOES} ---")

    # df.File(os.path.join(OUTPUT_DIR, f"debug/mesh_input_iter_{i+1}.pvd")) << X

    try:
        # coords_descarregada_atual = np.copy(mesh_atual.coordinates())
        mesh.coordinates()[:] = X.copy()
        mesh.bounding_box_tree().build(mesh)

        print("Executando simulação direta (chamando o solver)...")
        u_calculado = solve_inflation_lv(
            mesh, ffun, ldrb_markers, PRESSAO_MEDIDA, SOLVER_PRESSURE_STEPS 
        )
        u_calculado.rename("u", f"displacement_iter_{i+1}")
        disp_file << u_calculado
        # u_array = u_calculado.vector().get_local().reshape((-1, 3))
        # u_array = u_calculado.compute_vertex_values(mesh_atual)
        coords = mesh.coordinates()
        u_array = np.array([u_calculado(xx) for xx in coords])

        x = mesh.coordinates()[:].copy()
        x += u_at_vertices

        lres = []
        for i in range(np.shape(xm)[0]):
            lres.append(np.linalg.norm(xm[i,:]-x[i,:], 2))

        # coords_deformada_calculada = coords_descarregada_atual + u_array
        # residuo_vetorial = coords_medida - coords_deformada_calculada
        # max_residuo = np.max(np.linalg.norm(residuo_vetorial, axis=1))

        res = max(lres)
        
        residuos_por_iteracao.append(res)
        volume_calculado = compute_cavity_volume(mesh, ffun, ldrb_markers, u_calculado)
        volumes_por_iteracao.append(volume_calculado)
        
        print(f"Resíduo máximo nesta iteração: {res:.6f}")
        print(f"Volume da cavidade calculado: {volume_calculado:.2f}")

        if res < TOLERANCIA:
            print("\nConvergência atingida com sucesso!")
            break

        print("Atualizando a estimativa da geometria descarregada com sub-relaxação...")
        # target_coords_descarregada = coords_medida - u_array
        # correcao = target_coords_descarregada - coords_descarregada_atual
        # coords_descarregada_proxima = coords_descarregada_atual + FATOR_RELAXACAO * correcao
        # mesh_atual.coordinates()[:] = coords_descarregada_proxima

        X = xm.copy() - u_array

    except RuntimeError as e:
        print(f"\nERRO na simulação. O solver não convergiu na iteração {i+1}.")
        print("Detalhes do erro do FEniCS:", e)
        # mesh_atual.coordinates()[:] = coords_medida # remover
        break

else: 
    print(f"\nAVISO: Número máximo de {MAX_ITERACOES} iterações atingido sem convergência total.")

print(f"\nSalvando a geometria descarregada final em '{UNLOADED_MESH_FILE}'...")
with df.XDMFFile(UNLOADED_MESH_FILE) as outfile:
    outfile.write(mesh)
print("Processo concluído.")

# --- VISUALIZAÇÃO DOS RESULTADOS ---
if volumes_por_iteracao and residuos_por_iteracao:
    num_iteracoes = rangse(1, len(volumes_por_iteracao) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot do Resíduo (eixo Y esquerdo)
    color = 'tab:blue'
    ax1.set_xlabel('Iteração do Ponto Fixo')
    ax1.set_ylabel('Resíduo Máximo (Erro)', color=color)
    ax1.plot(num_iteracoes, residuos_por_iteracao, marker='o', linestyle='--', color=color, label='Resíduo Máximo')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log') # Escala logarítmica é ideal para ver a convergência do erro
    
    # Cria um segundo eixo Y que compartilha o mesmo eixo X
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Volume da Cavidade', color=color)
    ax2.plot(num_iteracoes, volumes_por_iteracao, marker='s', linestyle=':', color=color, label='Volume Calculado')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Convergência do Resíduo e do Volume por Iteração')
    plt.grid(True)
    plt.xticks(num_iteracoes)
    
    # Adiciona legendas de ambos os plots
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    plt.show()

