# -*- coding: utf-8 -*-

import dolfin as df
import numpy as np
import os
import matplotlib.pyplot as plt
from src.solver import solve_inflation_lvrv
from src.utils import compute_cavity_volume, plot_convergence_history

# --- 1. PARÂMETROS GERAIS DA SIMULAÇÃO ---
TOLERANCIA = 1e-3
MAX_ITERACOES = 100
PRESSAO_MEDIDA = [1.8, 0.8]  # Pressões alvo para [VE, VD]
SOLVER_PRESSURE_STEPS = 50

# --- NOVOS PARÂMETROS PARA DETECÇÃO DE ESTAGNAÇÃO ---
# Tolerância para a melhora relativa do erro. Se a redução do erro for menor que
# esta porcentagem, a iteração é considerada estagnada.
STAGNATION_TOLERANCE = 0.01  # 1%
# Número de iterações consecutivas de estagnação permitidas antes de parar.
STAGNATION_LIMIT = 10

# Caminhos de entrada e saída
MESH_PATH = "./data/ex2_lvrv/Patient_lvrv.xml"
FFUN_PATH = "./data/ex2_lvrv/Patient_lvrv_facet_region.xml"
OUTPUT_DIR = "results_unload/ex2_lvrv_teste1"
UNLOADED_MESH_FILE = os.path.join(OUTPUT_DIR, "unloaded_mesh.xdmf")
ITERATIVE_DISP_FILE = os.path.join(OUTPUT_DIR, "deslocamento_iterativo.pvd")
CONVERGENCE_GRAPH = os.path.join(OUTPUT_DIR, "convergence_graph.png")

# Dicionário de marcadores de fronteira
ldrb_markers = {"base": 10, "lv": 20, "epi": 40, "rv": 30}


# --- 2. INICIALIZAÇÃO DO ALGORITMO ---
df.set_log_level(df.LogLevel.WARNING)
print("Inicializando o processo para encontrar a geometria sem carga...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    mesh = df.Mesh(MESH_PATH)
    ffun = df.MeshFunction("size_t", mesh, FFUN_PATH)
except RuntimeError:
    print(f"\nERRO: Malha '{MESH_PATH}' ou fronteiras '{FFUN_PATH}' não encontradas.")
    exit()

xm = mesh.coordinates()[:].copy()
X = xm.copy()

disp_file = df.File(ITERATIVE_DISP_FILE)
volumes_por_iteracao = []
residuos_por_iteracao = []

# Variáveis para controle de estagnação
previous_res = float('inf') # Inicializa com um valor infinito
stagnation_counter = 0


# --- 3. LOOP ITERATIVO DE PONTO FIXO ---
for i in range(MAX_ITERACOES):
    print(f"\n--- Iteração {i+1}/{MAX_ITERACOES} ---")
    try:
        mesh.coordinates()[:] = X.copy()
        mesh.bounding_box_tree().build(mesh)
        # cells_colision = mesh.compute_entity_collisions(mesh)

        print("Executando simulação direta (chamando o solver)...")
        u_calculado, [fiber, sheet, sheet_normal] = solve_inflation_lvrv(
            mesh, ffun, ldrb_markers, PRESSAO_MEDIDA, SOLVER_PRESSURE_STEPS 
        )
        
        u_calculado.rename("u", f"displacement_iter_{i+1}")
        disp_file << u_calculado
        
        coords = mesh.coordinates()
        u_array = np.array([u_calculado(xx) for xx in coords])

        x = mesh.coordinates()[:].copy() + u_array
        
        # Cálculo do resíduo
        lres = [np.linalg.norm(xm[j,:]-x[j,:], 2) for j in range(np.shape(xm)[0])]
        res = max(lres)
        
        residuos_por_iteracao.append(res)
        volume_lv = compute_cavity_volume(mesh, ffun, ldrb_markers, "lv", u_calculado)
        volume_rv = compute_cavity_volume(mesh, ffun, ldrb_markers, "rv", u_calculado)
        volumes_por_iteracao.append([volume_lv, volume_rv])
        
        print(f"Resíduo máximo nesta iteração: {res:.6f}")

        # --- LÓGICA DE VERIFICAÇÃO DE CONVERGÊNCIA E ESTAGNAÇÃO ---
        if res < TOLERANCIA:
            print("\nConvergência atingida com sucesso!")
            X = mesh.coordinates()[:].copy()
            break

        # Verifica por estagnação ou divergência (apenas após a primeira iteração)
        if i > 0:
            # if res > previous_res:
            #     print("\nAVISO: O erro aumentou (divergência). Interrompendo a simulação.")
            #     break
            
            relative_change = abs(res - previous_res) / previous_res
            if relative_change < STAGNATION_TOLERANCE:
                stagnation_counter += 1
                print(f"  > Progresso de convergência lento (estagnação: {stagnation_counter}/{STAGNATION_LIMIT}).")
                if stagnation_counter >= STAGNATION_LIMIT:
                    print("\nAVISO: Limite de estagnação atingido. Interrompendo a simulação.")
                    break
            else:
                # Se houve progresso, reseta o contador
                stagnation_counter = 0
        
        previous_res = res # Atualiza o resíduo da iteração anterior
        # --- FIM DA LÓGICA DE VERIFICAÇÃO ---

        # Atualização da geometria para a próxima iteração
        print("Atualizando a estimativa da geometria descarregada...")
        X = xm.copy() - u_array

    except RuntimeError as e:
        print(f"\nERRO na simulação na iteração {i+1}. O solver não convergiu.")
        print("Detalhes do erro do FEniCS:", e)
        break
else: 
    print(f"\nAVISO: Número máximo de {MAX_ITERACOES} iterações atingido sem convergência total.")

# --- 4. SALVAMENTO E VISUALIZAÇÃO ---
mesh.coordinates()[:] = X

print(f"\nSalvando a geometria descarregada final em '{UNLOADED_MESH_FILE}'...")
with df.XDMFFile(UNLOADED_MESH_FILE) as outfile:
    outfile.write(mesh)
    fiber.rename("f", "f")
    sheet.rename("s", "s")
    sheet_normal.rename("n", "n")
    outfile.write(fiber, 0)
    outfile.write(sheet, 0)
    outfile.write(sheet_normal, 0)
print("Processo concluído.")

if volumes_por_iteracao and residuos_por_iteracao:
    plot_convergence_history(volumes_por_iteracao, residuos_por_iteracao, CONVERGENCE_GRAPH)

