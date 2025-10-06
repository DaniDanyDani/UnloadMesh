# -*- coding: utf-8 -*-
from dolfin import *
import numpy as np
from src.guccionematerial import GuccioneMaterial
import ldrb

def solve_inflation_lv(mesh, ffun, ldrb_markers, pressure_value = 1.8, num_steps=50):
    """
    Executa uma simulação de inflação passiva, aplicando a carga em `num_steps` incrementos.
    
    Argumentos:
        mesh (dolfin.Mesh): A malha na configuração de referência.
        ffun (dolfin.MeshFunction): Os marcadores de fronteira.
        ldrb_markers (dict): Dicionário com os IDs dos marcadores.
        pressure_value (float): O valor da pressão FINAL a ser aplicado.
        num_steps (int): O número de passos para aplicar a carga.
        
    Retorna:
        dolfin.Function: O campo de deslocamento 'u' resultante na pressão final.
    """
    
    parameters["form_compiler"]["quadrature_degree"] = 4
    parameters["form_compiler"]["cpp_optimize"] = True
    
    # --- CÁLCULO DAS FIBRAS ---
    fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(
        mesh=mesh, fiber_space="DG_0", ffun=ffun, markers=ldrb_markers,
        alpha_endo_lv=30, alpha_epi_lv=-30, beta_endo_lv=0.0, beta_epi_lv=0.0,
        alpha_endo_sept=60.0, alpha_epi_sept=-60.0, beta_endo_sept=0.0, beta_epi_sept=0.0,
        alpha_endo_rv=80.0, alpha_epi_rv=-80.0, beta_endo_rv=0.0, beta_epi_rv=0.0
    )

    # --- CONFIGURAÇÃO DO PROBLEMA DE ELEMENTOS FINITOS ---
    V = VectorFunctionSpace(mesh, 'P', 1)
    ds = Measure('ds', domain=mesh, subdomain_data=ffun)

    clamp = Constant((0.0, 0.0, 0.0))
    bc = DirichletBC(V, clamp, ffun, ldrb_markers["base"])
    bcs = [bc]

    u = Function(V)
    v = TestFunction(V)

    I = Identity(3); F = I + grad(u); F = variable(F)
    mat = GuccioneMaterial(e1=fiber, e2=sheet, e3=sheet_normal, kappa=1e3, Tactive=0.0)
    psi = mat.strain_energy(F)
    P = diff(psi, F)
    
    # A pressão agora é uma constante que será atualizada dentro do loop
    p_endo = Constant(0.0)
    
    N = FacetNormal(mesh)
    Gext = p_endo * inner(v, det(F) * inv(F) * N) * ds(ldrb_markers["lv"])
    R = inner(P, grad(v)) * dx + Gext

    newton_solver_params = {"relative_tolerance": 1e-5, 
                            "maximum_iterations": 50, 
                            "linear_solver": 'mumps', 
                            "absolute_tolerance": 1e-5,
                            "relaxation_parameter": 0.7}
    
    solver_params = {"nonlinear_solver": 'newton',
                     "newton_solver": newton_solver_params}
    
    # --- APLICAÇÃO DA CARGA INCREMENTAL ---
    print(f"  Aplicando pressão {pressure_value:.2f} em {num_steps} passo(s)...")
    
    # Gera os valores de pressão para cada passo
    # Ex: para 10 kPa em 5 passos -> [2.0, 4.0, 6.0, 8.0, 10.0]
    # pressures = np.linspace(pressure_value / num_steps, pressure_value, num=num_steps)
    pressures = np.linspace(0, pressure_value, num=num_steps)

    # O campo 'u' começa em zero e é atualizado a cada passo, usando a solução anterior
    # como chute inicial para a próxima, o que melhora a convergência.
    for i, p in enumerate(pressures):
        print(f"    Passo de carga {i+1}/{num_steps}, Pressão = {p:.2f}")
        p_endo.assign(p)
        
        solve(R == 0, u, bcs, solver_parameters=solver_params)
          
    # Retorna o campo de deslocamento final
    return u, [fiber, sheet, sheet_normal]

def solve_inflation_lvrv(mesh, ffun, ldrb_markers, pressure_value = [1.8, 1], num_steps=50):
    """
    Executa uma simulação de inflação passiva para uma malha biventricular.
    
    Argumentos:
        mesh (dolfin.Mesh): A malha na configuração de referência biventricular.
        ffun (dolfin.MeshFunction): Os marcadores de fronteira.
        ldrb_markers (dict): Dicionário com os IDs dos marcadores.
        pressure_value (list, float): O valor da pressão FINAL a ser aplicado no lv e rv, respectivamente.
        num_steps (int): O número de passos para aplicar a carga.
        
    Retorna:
        dolfin.Function: O campo de deslocamento 'u' resultante na pressão final.
    """
    
    parameters["form_compiler"]["quadrature_degree"] = 4
    parameters["form_compiler"]["cpp_optimize"] = True
    
    # --- CÁLCULO DAS FIBRAS ---
    fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(
        mesh=mesh, fiber_space="DG_0", ffun=ffun, markers=ldrb_markers,
        alpha_endo_lv=30, alpha_epi_lv=-30, beta_endo_lv=0.0, beta_epi_lv=0.0,
        alpha_endo_sept=60.0, alpha_epi_sept=-60.0, beta_endo_sept=0.0, beta_epi_sept=0.0,
        alpha_endo_rv=80.0, alpha_epi_rv=-80.0, beta_endo_rv=0.0, beta_epi_rv=0.0
    )

    # --- CONFIGURAÇÃO DO PROBLEMA DE ELEMENTOS FINITOS ---
    V = VectorFunctionSpace(mesh, 'P', 1)
    ds = Measure('ds', domain=mesh, subdomain_data=ffun)

    clamp = Constant((0.0, 0.0, 0.0))
    bc = DirichletBC(V, clamp, ffun, ldrb_markers["base"])
    bcs = [bc]

    u = Function(V)
    v = TestFunction(V)

    I = Identity(3); F = I + grad(u); F = variable(F)
    mat = GuccioneMaterial(e1=fiber, e2=sheet, e3=sheet_normal, kappa=1e3, Tactive=0.0)
    psi = mat.strain_energy(F)
    P = diff(psi, F)
    
    # A pressão agora é uma constante que será atualizada dentro do loop
    p_endo = Constant(0.0)
    
    N = FacetNormal(mesh)
    Gext = p_endo * inner(v, det(F) * inv(F) * N) * ds(ldrb_markers["lv"])
    R = inner(P, grad(v)) * dx + Gext

    newton_solver_params = {"relative_tolerance": 1e-5, 
                            "maximum_iterations": 50, 
                            "linear_solver": 'mumps', 
                            "absolute_tolerance": 1e-5,
                            "relaxation_parameter": 0.7}
    
    solver_params = {"nonlinear_solver": 'newton',
                     "newton_solver": newton_solver_params}
    
    # --- APLICAÇÃO DA CARGA INCREMENTAL ---
    print(f"  Aplicando pressão {pressure_value:.2f} em {num_steps} passo(s)...")
    
    # Gera os valores de pressão para cada passo
    # Ex: para 10 kPa em 5 passos -> [2.0, 4.0, 6.0, 8.0, 10.0]
    # pressures = np.linspace(pressure_value / num_steps, pressure_value, num=num_steps)
    pressures = np.linspace(0, pressure_value, num=num_steps)

    # O campo 'u' começa em zero e é atualizado a cada passo, usando a solução anterior
    # como chute inicial para a próxima, o que melhora a convergência.
    for i, p in enumerate(pressures):
        print(f"    Passo de carga {i+1}/{num_steps}, Pressão = {p:.2f}")
        p_endo.assign(p)
        
        solve(R == 0, u, bcs, solver_parameters=solver_params)
          
    # Retorna o campo de deslocamento final
    return u, [fiber, sheet, sheet_normal]

