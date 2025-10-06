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

# --- 1. PARÂMETROS GERAIS ---
TOLERANCIA = 1e-5      
MAX_ITERACOES = 100000     
PRESSAO_MEDIDA = 10.0  
SOLVER_PRESSURE_STEPS = 500

FATOR_RELAXACAO = 0.01

# Caminhos dos arquivos
MESH_PATH = "./data/example/Patient_lv.xml"
FFUN_PATH = "./data/example/Patient_lv_facet_region.xml"
OUTPUT_DIR = "results_unload/teste_6"
UNLOADED_MESH_FILE = os.path.join(OUTPUT_DIR, "unloaded_mesh.xdmf")
ITERATIVE_DISP_FILE = os.path.join(OUTPUT_DIR, "deslocamento_iterativo.pvd")

# Marcações da malha
ldrb_markers = {"base": 10, "lv": 20, "epi": 40, "rv": 30}


# --- 2. INICIALIZAÇÃO DO ALGORITMO ---
print("Inicializando o processo para encontrar a geometria sem carga...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    mesh_atual = df.Mesh(MESH_PATH)
    ffun = df.MeshFunction("size_t", mesh_atual, FFUN_PATH)
    coords_medida = np.copy(mesh_atual.coordinates()) 
except RuntimeError:
    print(f"\nERRO: Malha '{MESH_PATH}' ou fronteiras '{FFUN_PATH}' não encontradas.")
    exit()

disp_file = df.File(ITERATIVE_DISP_FILE)
# Listas para armazenar os resultados de cada iteração para plotagem
volumes_por_iteracao = []
residuos_por_iteracao = []


# --- 3. LOOP ITERATIVO DE PONTO FIXO ---
for i in range(MAX_ITERACOES):
    print(f"\n--- Iteração {i+1}/{MAX_ITERACOES} ---")

    try:
        coords_descarregada_atual = np.copy(mesh_atual.coordinates())

        # PASSO 1: SIMULAÇÃO DIRETA
        print("Executando simulação direta (chamando o solver)...")
        u_calculado = solve_inflation_lv(
            mesh_atual, ffun, ldrb_markers, PRESSAO_MEDIDA, SOLVER_PRESSURE_STEPS 
        )
        u_calculado.rename("u", f"displacement_iter_{i+1}")
        disp_file << u_calculado
        u_array = u_calculado.vector().get_local().reshape((-1, 3))

        # PASSO 2: CÁLCULO DO RESÍDUO
        coords_deformada_calculada = coords_descarregada_atual + u_array
        residuo_vetorial = coords_medida - coords_deformada_calculada
        max_residuo = np.max(np.linalg.norm(residuo_vetorial, axis=1))
        
        # Armazena os resultados para plotagem
        residuos_por_iteracao.append(max_residuo)
        volume_calculado = compute_cavity_volume(mesh_atual, ffun, ldrb_markers, u_calculado)
        volumes_por_iteracao.append(volume_calculado)
        
        print(f"Resíduo máximo nesta iteração: {max_residuo:.6f}")
        print(f"Volume da cavidade calculado: {volume_calculado:.2f}")

        if max_residuo < TOLERANCIA:
            print("\nConvergência atingida com sucesso!")
            break

        # PASSO 3: ATUALIZAÇÃO DA GEOMETRIA COM SUB-RELAXAÇÃO
        print("Atualizando a estimativa da geometria descarregada com sub-relaxação...")
        target_coords_descarregada = coords_medida - u_array
        correcao = target_coords_descarregada - coords_descarregada_atual
        coords_descarregada_proxima = coords_descarregada_atual + FATOR_RELAXACAO * correcao
        mesh_atual.coordinates()[:] = coords_descarregada_proxima

    except RuntimeError as e:
        print(f"\nERRO na simulação. O solver não convergiu na iteração {i+1}.")
        print("Detalhes do erro do FEniCS:", e)
        # mesh_atual.coordinates()[:] = coords_medida # remover
        break

else: 
    print(f"\nAVISO: Número máximo de {MAX_ITERACOES} iterações atingido sem convergência total.")

# --- 4. SALVAR RESULTADO FINAL ---
print(f"\nSalvando a geometria descarregada final em '{UNLOADED_MESH_FILE}'...")
with df.XDMFFile(UNLOADED_MESH_FILE) as outfile:
    outfile.write(mesh_atual)
print("Processo concluído.")

# --- 5. VISUALIZAÇÃO DOS RESULTADOS ---
if volumes_por_iteracao and residuos_por_iteracao:
    num_iteracoes = range(1, len(volumes_por_iteracao) + 1)

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

