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

    # Usa o marcador 'lv' definido no dicionário ldrb_markers
    return df.assemble(vol_form*ds(numbering["lv"]))

# --- 1. PARÂMETROS GERAIS ---
# Parâmetros do Algoritmo de Ponto Fixo
TOLERANCIA = 1e-3      # Critério de parada (epsilon do artigo)
MAX_ITERACOES = 100     # Número máximo de iterações para o ponto fixo.
PRESSAO_MEDIDA = 1.0  # Pressão alvo 'pm' do artigo.
SOLVER_PRESSURE_STEPS = 100 # Em quantos passos o solver interno deve dividir a carga.

# Caminhos dos arquivos
MESH_PATH = "./data/example/Patient_lv.xml"
FFUN_PATH = "./data/example/Patient_lv_facet_region.xml"
OUTPUT_DIR = "results_unload/teste_3"
UNLOADED_MESH_FILE = os.path.join(OUTPUT_DIR, "unloaded_mesh.xdmf")
ITERATIVE_DISP_FILE = os.path.join(OUTPUT_DIR, "deslocamento_iterativo.pvd")

# Marcações da malha
ldrb_markers = {"base": 10, "lv": 20, "epi": 40, "rv": 30}


# --- 2. INICIALIZAÇÃO DO ALGORITMO ---
print("Inicializando o processo para encontrar a geometria sem carga...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    # Carrega a geometria medida 'xm', que está sob pressão.
    mesh_atual = df.Mesh(MESH_PATH)
    ffun = df.MeshFunction("size_t", mesh_atual, FFUN_PATH)
    # Salva as coordenadas originais 'xm' para referência.
    coords_medida = np.copy(mesh_atual.coordinates()) 
except RuntimeError:
    print(f"\nERRO: Malha '{MESH_PATH}' ou fronteiras '{FFUN_PATH}' não encontradas.")
    exit()

# Passo 2 do Algoritmo 1: O chute inicial para a geometria descarregada ('X_1')
# é a própria geometria medida ('xm').
# A variável 'mesh_atual' representa 'X_i' em cada iteração.
disp_file = df.File(ITERATIVE_DISP_FILE)

# Lista para armazenar o volume calculado a cada iteração
volumes_por_iteracao = []

# disp_file = File("results_unload/teste_50intPF/u.pvd")

# --- 3. LOOP ITERATIVO DE PONTO FIXO ---
# Passo 3 do Algoritmo 1: Inicia o loop 'while'.
for i in range(MAX_ITERACOES):
    print(f"\n--- Iteração {i+1}/{MAX_ITERACOES} ---")

    try:
        # Salva as coordenadas da estimativa atual da malha descarregada ('X_i')
        coords_descarregada_atual = np.copy(mesh_atual.coordinates())

        # =========================================================================
        # PASSO 1: SIMULAÇÃO DIRETA (Linha 5 do Algoritmo 1)
        # Calcula a configuração de equilíbrio Omega(x_i, sigma_i) a partir da
        # estimativa atual da geometria descarregada Omega(X_i, 0) sob a pressão 'pm'.
        # O resultado principal é o campo de deslocamento 'U_i = x_i - X_i'.
        # =========================================================================
        print("Executando simulação direta (chamando o solver)...")
        u_calculado = solve_inflation_lv(
            mesh_atual, 
            ffun,       
            ldrb_markers,
            PRESSAO_MEDIDA,
            SOLVER_PRESSURE_STEPS 
        )
        u_calculado.rename("u", f"displacement_iter_{i+1}")
        disp_file << u_calculado
        u_array = u_calculado.vector().get_local().reshape((-1, 3))

        # =========================================================================
        # PASSO 2: CÁLCULO DO RESÍDUO (Linha 3 do Algoritmo 1)
        # Verifica a condição de parada. O resíduo 'r' é a distância entre
        # a geometria alvo 'xm' e a geometria deformada calculada na simulação 'x_i'.
        # O loop para quando max(r_i) < epsilon.
        # =========================================================================
        coords_deformada_calculada = coords_descarregada_atual + u_array
        residuo_vetorial = coords_medida - coords_deformada_calculada
        max_residuo = np.max(np.linalg.norm(residuo_vetorial, axis=1))
        
        print(f"Resíduo máximo nesta iteração: {max_residuo:.6f}")

        # Calcula e armazena o volume da cavidade deformada
        volume_calculado = compute_cavity_volume(mesh_atual, ffun, ldrb_markers, u_calculado)
        volumes_por_iteracao.append(volume_calculado)
        print(f"Volume da cavidade calculado: {volume_calculado:.2f}")

        if max_residuo < TOLERANCIA:
            print("\nConvergência atingida com sucesso!")
            break

        # =========================================================================
        # PASSO 3: ATUALIZAÇÃO DA GEOMETRIA (Linha 7 do Algoritmo 1)
        # Calcula a próxima estimativa da geometria descarregada usando a fórmula
        # de ponto fixo: X_{i+1} = x_m - U_i.
        # =========================================================================
        print("Atualizando a estimativa da geometria descarregada para a próxima iteração...")
        coords_descarregada_proxima = coords_medida - u_array
        mesh_atual.coordinates()[:] = coords_descarregada_proxima

    except RuntimeError as e:
        print(f"\nERRO na simulação. O solver não convergiu na iteração {i+1}.")
        print("Detalhes do erro do FEniCS:", e)
        mesh_atual.coordinates()[:] = coords_medida
        break

else: # Executado se o loop 'for' terminar sem um 'break'.
    print(f"\nAVISO: Número máximo de {MAX_ITERACOES} iterações atingido sem convergência total.")


# --- 4. SALVAR RESULTADO FINAL ---
# Linha 9 do Algoritmo 1: A geometria descarregada final 'X*' é a última estimativa 'X_i'.
print(f"\nSalvando a geometria descarregada final em '{UNLOADED_MESH_FILE}'...")
with df.XDMFFile(UNLOADED_MESH_FILE) as outfile:
    outfile.write(mesh_atual)

print("Processo concluído.")

# --- 5. VISUALIZAÇÃO DOS RESULTADOS ---
# Plota o gráfico da convergência do volume ao longo das iterações.
if volumes_por_iteracao:
    plt.figure()
    plt.plot(range(1, len(volumes_por_iteracao) + 1), volumes_por_iteracao, marker='o', linestyle='--')
    plt.xlabel("Iteração do Ponto Fixo")
    plt.ylabel("Volume da Cavidade Calculado")
    plt.title("Convergência do Volume da Cavidade por Iteração")
    plt.grid(True)
    plt.xticks(range(1, len(volumes_por_iteracao) + 1)) # Garante ticks inteiros no eixo x
    plt.show()

