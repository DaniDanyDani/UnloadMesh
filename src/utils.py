# -*- coding: utf-8 -*-
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_cavity_volume(mesh, mf, numbering, cavity_name = "lv", u=None):
    """
    Calcula o volume da cavidade usando uma integral de superfície no endocárdio.
    """
    X = SpatialCoordinate(mesh) 
    N = FacetNormal(mesh)

    if u is not None:
        I = Identity(3)
        F = I + grad(u)
        J = det(F)
        vol_form = (-1.0/3.0) * dot(X + u, J * inv(F).T * N)
    else:
        vol_form = (-1.0/3.0) * dot(X, N)

    ds = Measure('ds', domain=mesh, subdomain_data=mf)
    return assemble(vol_form*ds(numbering[cavity_name]))


def plot_convergence_history(volumes, residuos, output_path):
    """
    Gera e salva um gráfico mostrando a convergência do resíduo e a evolução do volume.

    Argumentos:
        volumes (list): Lista com o volume da cavidade a cada iteração.
        residuos (list): Lista com o resíduo máximo a cada iteração.
        output_path (str): Caminho completo para salvar o arquivo de imagem do gráfico.
    """
    if not volumes or not residuos:
        print("AVISO: Listas de volume ou resíduo vazias. O gráfico não será gerado.")
        return

    num_iteracoes = range(1, len(residuos) + 1)

    volumes_arr = np.array(volumes)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot do Resíduo (eixo Y esquerdo)
    color = 'tab:blue'
    ax1.set_xlabel('Iteração do Ponto Fixo', fontsize=12)
    ax1.set_ylabel('Resíduo Máximo (Erro)', color=color, fontsize=12)
    ax1.plot(num_iteracoes, residuos, marker='o', linestyle='--', color=color, label='Resíduo Máximo')
    ax1.tick_params(axis='y', labelcolor=color)
    # Escala logarítmica é ideal para ver a convergência do erro
    ax1.set_yscale('log') 
    
    # Cria um segundo eixo Y que compartilha o mesmo eixo X
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Volume da Cavidade Calculado', color=color, fontsize=12)

    if volumes_arr.ndim == 2 and volumes_arr.shape[1] == 2:
        volumes_lv = volumes_arr[:,0]
        volumes_rv = volumes_arr[:,1]
        ax2.plot(num_iteracoes, volumes_lv, marker='s', linestyle=':', color='tab:red', label='Volume LV')
        ax2.plot(num_iteracoes, volumes_rv, marker='^', linestyle=':', color='tab:green', label='Volume RV')
        ax2.tick_params(axis='y')
    else:
        color = 'tab:red'
        ax2.plot(num_iteracoes, volumes, marker='s', linestyle=':', color=color, label='Volume Calculado')
        ax2.tick_params(axis='y', labelcolor=color)


    fig.tight_layout(rect=[0, 0, 1, 0.96]) # Ajusta o layout para dar espaço ao título
    plt.title('Convergência do Resíduo e do Volume por Iteração', fontsize=14, pad=20)
    plt.grid(True)
    plt.xticks(num_iteracoes)
    
    # Adiciona legendas de ambos os plots
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    
    # Salva e mostra o gráfico
    plt.savefig(output_path)
    print(f"Gráfico de convergência salvo em '{output_path}'")
    plt.show()
