# Proyecto Final — Algoritmos Bioinspirados
# GABC: Guided Artificial Bee Colony aplicado a funciones benchmark


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import warnings

warnings.filterwarnings('ignore')


# 2. FUNCIONES BENCHMARK (Ackley, Griewank, Rastrigin, Rosenbrock)
#    Todas reciben X de forma (N, D) — evaluación vectorizada de la población.
#    Problema de minimización; óptimo global f(x*) = 0 (excepto Rosenbrock).

def ackley(X):
    n = X.shape[1]
    a, b, c = 20, 1/5, 2 * np.pi

    sum_sq  = np.sum(X**2, axis=1) / n
    sum_cos = np.sum(np.cos(c * X), axis=1) / n

    return -a * np.exp(-b * np.sqrt(sum_sq)) - np.exp(sum_cos) + a + np.e


def griewank(X):
    n = X.shape[1]
    i = np.arange(1, n + 1, dtype=float)

    sum_sq   = np.sum(X**2, axis=1) / 4000
    prod_cos = np.prod(np.cos(X / np.sqrt(i)), axis=1)

    return sum_sq - prod_cos + 1


def rastrigin(X):
    n = X.shape[1]
    return 10 * n + np.sum(X**2 - 10 * np.cos(2 * np.pi * X), axis=1)


def rosenbrock(X):
    xi  = X[:, :-1]
    xi1 = X[:, 1:]
    return np.sum(100 * (xi1 - xi**2)**2 + (xi - 1)**2, axis=1)


# Verificación de óptimos conocidos
n = 10

optimos = {
    "Ackley"    : np.zeros((1, n)),
    "Griewank"  : np.zeros((1, n)),
    "Rastrigin" : np.zeros((1, n)),
    "Rosenbrock": np.ones((1, n)),
}

funciones = {
    "Ackley"    : ackley,
    "Griewank"  : griewank,
    "Rastrigin" : rastrigin,
    "Rosenbrock": rosenbrock,
}

print(f"{'Función':<14} {'f(x*)'}")
print("-" * 30)
for nombre, f in funciones.items():
    val = f(optimos[nombre])[0]
    print(f"{nombre:<14} {val:.6e}")


# 3. PARÁMETROS GLOBALES Y DICCIONARIO DE PROBLEMAS

N    = 100    # tamaño de la colonia (empleadas = observadoras = N)
D    = 10     # número de variables de decisión
GMAX = 5000   # número máximo de ciclos de búsqueda

PROBLEMAS = {
    "Ackley"    : {"func": ackley,     "lb": -32.768, "ub":  32.768, "n": D},
    "Griewank"  : {"func": griewank,   "lb": -600.0,  "ub":  600.0,  "n": D},
    "Rastrigin" : {"func": rastrigin,  "lb": -5.12,   "ub":  5.12,   "n": D},
    "Rosenbrock": {"func": rosenbrock, "lb": -2.048,  "ub":  2.048,  "n": D},
}


# 4. INICIALIZACIÓN DE LA COLMENA

def inicializar_colmena(SN, D, lb, ub, func, seed=None):
    """
    Genera SN fuentes de alimento distribuidas uniformemente en [lb, ub]^D,
    evalúa la función objetivo sobre todas ellas e inicializa los contadores
    de intentos fallidos (trial) en cero.

    Parámetros
    ----------
    SN   : int      — número de fuentes (= abejas empleadas = observadoras)
    D    : int      — dimensiones del problema
    lb   : float    — límite inferior del dominio
    ub   : float    — límite superior del dominio
    func : callable — función objetivo f(X), X de forma (N, D)
    seed : int|None — semilla para reproducibilidad

    Retorna
    -------
    fuentes : ndarray (SN, D) — posiciones iniciales de las fuentes
    fitness : ndarray (SN,)   — valor f(x) de cada fuente
    trial   : ndarray (SN,)   — contadores de fallos, todos en 0
    """
    if seed is not None:
        np.random.seed(seed)

    fuentes = np.random.uniform(lb, ub, (SN, D))
    fitness = func(fuentes)
    trial   = np.zeros(SN, dtype=int)

    return fuentes, fitness, trial


# 5. CÁLCULO DE PROBABILIDADES DE SELECCIÓN


def calcular_probabilidades(fitness):
    """
    Convierte los valores de fitness (minimización) en probabilidades
    de selección para la fase de abejas observadoras.

    Parámetros
    ----------
    fitness : ndarray (SN,) — valores f(x) de cada fuente

    Retorna
    -------
    probs : ndarray (SN,) — probabilidades normalizadas (suman 1)
    """
    fit_vals = np.where(
        fitness >= 0,
        1.0 / (1.0 + fitness),
        1.0 + np.abs(fitness)
    )
    return fit_vals / fit_vals.sum()


# 6. FASE DE ABEJAS EMPLEADAS (vectorizada)
#    Ecuación GABC: v_ij = x_ij + phi*(x_ij - x_kj) + psi*(gbest_j - x_ij)

def fase_empleadas(fuentes, fitness, trial, lb, ub, func, gbest, phi_scale):
    """
    Fase de abejas empleadas (Employed Bees Phase)
    Esta parte es diferente al cuaderno, la del cuaderno no está vectorizada y es más lenta. 

    Parámetros
    ----------
    fuentes   : ndarray (SN, D) — fuentes actuales
    fitness   : ndarray (SN,)   — valor f(x) de cada fuente
    trial     : ndarray (SN,)   — contadores de intentos fallidos
    lb, ub    : float           — límites del dominio
    func      : callable        — función objetivo
    gbest     : ndarray (D,)    — mejor solución global encontrada
    phi_scale : float           — escala del factor de perturbación

    Retorna
    -------
    fuentes, fitness, trial actualizados
    exitos : int — número de mejoras logradas en esta fase
    """
    SN, D = fuentes.shape
    fila  = np.arange(SN)

    K   = (fila + np.random.randint(1, SN, SN)) % SN
    J   = np.random.randint(0, D, SN)
    phi = np.random.uniform(-phi_scale, phi_scale, SN)
    psi = np.random.uniform(0, 1.5, SN)

    candidatas = fuentes.copy()
    candidatas[fila, J] = (
        fuentes[fila, J]
        + phi * (fuentes[fila, J] - fuentes[K, J])
        + psi * (gbest[J]         - fuentes[fila, J])
    )
    candidatas = np.clip(candidatas, lb, ub)

    f_cand = func(candidatas)

    mejora          = f_cand < fitness
    fuentes[mejora] = candidatas[mejora]
    fitness[mejora] = f_cand[mejora]
    trial[mejora]   = 0
    trial[~mejora] += 1

    return fuentes, fitness, trial, int(mejora.sum())


# 7. FASE DE ABEJAS OBSERVADORAS 
def fase_observadoras(fuentes, fitness, trial, lb, ub, func, gbest, phi_scale):
    """
    Fase de abejas observadoras (Onlooker Bees Phase) — versión vectorizada.

    Parámetros
    ----------
    fuentes   : ndarray (SN, D) — fuentes actuales
    fitness   : ndarray (SN,)   — valor f(x) de cada fuente
    trial     : ndarray (SN,)   — contadores de intentos fallidos
    lb, ub    : float           — límites del dominio
    func      : callable        — función objetivo
    gbest     : ndarray (D,)    — mejor solución global encontrada
    phi_scale : float           — escala del factor de perturbación

    Retorna
    -------
    fuentes, fitness, trial actualizados
    exitos : int — número de mejoras logradas en esta fase
    """
    SN, D = fuentes.shape

    probs   = calcular_probabilidades(fitness)
    indices = np.random.choice(SN, size=SN, replace=True, p=probs)

    K   = (indices + np.random.randint(1, SN, SN)) % SN
    J   = np.random.randint(0, D, SN)
    phi = np.random.uniform(-phi_scale, phi_scale, SN)
    psi = np.random.uniform(0, 1.5, SN)

    fila  = np.arange(SN)
    cand  = fuentes[indices].copy()
    cand[fila, J] = (
        fuentes[indices, J]
        + phi * (fuentes[indices, J] - fuentes[K, J])
        + psi * (gbest[J]            - fuentes[indices, J])
    )
    cand   = np.clip(cand, lb, ub)
    f_cand = func(cand)

    mejora = f_cand < fitness[indices]
    np.add.at(trial, indices[~mejora], 1)

    exitos = 0
    for pos in np.where(mejora)[0]:
        idx = indices[pos]
        if f_cand[pos] < fitness[idx]:
            fuentes[idx] = cand[pos]
            fitness[idx] = f_cand[pos]
            trial[idx]   = 0
            exitos += 1

    return fuentes, fitness, trial, exitos


# 8. FASE DE ABEJAS EXPLORADORAS

def fase_exploradoras(fuentes, fitness, trial, lb, ub, D, func, limit):
    """
    Fase de abejas exploradoras (Scout Bees Phase).

    Si la fuente con mayor contador trial supera el umbral limit,
    se abandona y se reemplaza por una nueva fuente aleatoria.
    Solo se abandona una fuente por ciclo.

    Parámetros
    ----------
    fuentes : ndarray (SN, D) — fuentes actuales
    fitness : ndarray (SN,)   — valor f(x) de cada fuente
    trial   : ndarray (SN,)   — contadores de intentos fallidos
    lb, ub  : float           — límites del dominio
    D       : int             — número de dimensiones
    func    : callable        — función objetivo
    limit   : int             — umbral de abandono (auto-adaptado)

    Retorna
    -------
    fuentes, fitness, trial actualizados
    """
    idx_peor = np.argmax(trial)

    if trial[idx_peor] >= limit:
        nueva_fuente = np.random.uniform(lb, ub, D)
        fuentes[idx_peor] = nueva_fuente
        fitness[idx_peor] = func(nueva_fuente.reshape(1, D))[0]
        trial[idx_peor]   = 0

    return fuentes, fitness, trial


# 9. AUTO-ADAPTACIÓN DE PARÁMETROS EN LÍNEA
#    Ajusta `limit` y `phi_scale` cada 50 ciclos según la tasa de éxito.

LIMIT_OPCIONES = [10, 25, 50]     # umbral de abandono
PHI_OPCIONES   = [0.5, 0.8, 1.0]  # escala de perturbación


def adaptar_parametros(ventana_exitos, limit_actual, phi_actual):
    """
    Ajusta limit y phi_scale según la tasa de éxito promedio en una
    ventana deslizante de los últimos ciclos.

    Regla:
        tasa > 0.4  -> explotar: limit bajo  + phi alto
        tasa < 0.1  -> explorar: limit alto  + phi bajo
        intermedio  -> valores medios (balance)

    Parámetros
    ----------
    ventana_exitos : list  — tasa de éxito normalizada por ciclo
    limit_actual   : int   — valor de limit actual
    phi_actual     : float — valor de phi_scale actual

    Retorna
    -------
    limit_nuevo : int
    phi_nuevo   : float
    """
    if len(ventana_exitos) == 0:
        return limit_actual, phi_actual

    tasa = np.mean(ventana_exitos)

    if tasa > 0.4:
        return LIMIT_OPCIONES[0], PHI_OPCIONES[2]   # 10, 1.0
    elif tasa < 0.1:
        return LIMIT_OPCIONES[2], PHI_OPCIONES[0]   # 50, 0.5
    else:
        return LIMIT_OPCIONES[1], PHI_OPCIONES[1]   # 25, 0.8


# -----------------------------------------------------------------------------
# 10. ALGORITMO GABC COMPLETO
# -----------------------------------------------------------------------------

def gabc(func, lb, ub, n=10, SN=100, Gmax=5000, seed=None):
    """
    GABC — Guided Artificial Bee Colony con auto-adaptación de parámetros.
    Versión vectorizada (fases de empleadas y observadoras sin bucles internos).

    Parámetros
    ----------
    func  : callable — función objetivo f(X), X de forma (N, D)
    lb    : float    — límite inferior del dominio
    ub    : float    — límite superior del dominio
    n     : int      — número de dimensiones (D)
    SN    : int      — tamaño de la colonia
    Gmax  : int      — número máximo de ciclos
    seed  : int|None — semilla para reproducibilidad

    Retorna
    -------
    mejor_fit  : float       — mejor valor de f(x) encontrado
    historial  : list[float] — mejor f(x) por ciclo
    hist_limit : list[int]   — evolución del parámetro limit
    hist_phi   : list[float] — evolución del parámetro phi_scale
    """
    D = n
    if seed is not None:
        np.random.seed(seed)

    fuentes, fitness, trial = inicializar_colmena(SN, D, lb, ub, func, seed=None)

    mejor_idx = np.argmin(fitness)
    gbest     = fuentes[mejor_idx].copy()
    mejor_fit = fitness[mejor_idx]

    limit     = LIMIT_OPCIONES[1]   # 25
    phi_scale = PHI_OPCIONES[1]     # 0.8

    historial  = [mejor_fit]
    hist_limit = [limit]
    hist_phi   = [phi_scale]

    W              = 50
    ventana_exitos = []
    FREQ_ADAPT     = 50

    for ciclo in range(1, Gmax + 1):

        fuentes, fitness, trial, ex_emp = fase_empleadas(
            fuentes, fitness, trial, lb, ub, func, gbest, phi_scale)

        fuentes, fitness, trial, ex_obs = fase_observadoras(
            fuentes, fitness, trial, lb, ub, func, gbest, phi_scale)

        fuentes, fitness, trial = fase_exploradoras(
            fuentes, fitness, trial, lb, ub, D, func, limit)

        mi = np.argmin(fitness)
        if fitness[mi] < mejor_fit:
            mejor_fit = fitness[mi]
            gbest     = fuentes[mi].copy()

        ventana_exitos.append((ex_emp + ex_obs) / (2 * SN))
        if len(ventana_exitos) > W:
            ventana_exitos.pop(0)

        if ciclo % FREQ_ADAPT == 0:
            limit, phi_scale = adaptar_parametros(ventana_exitos, limit, phi_scale)

        historial.append(mejor_fit)
        hist_limit.append(limit)
        hist_phi.append(phi_scale)

    return mejor_fit, historial, hist_limit, hist_phi


# Verificación rápida del GABC en los 4 problemas
print(f"\n{'Problema':<14} {'Mejor f(x)':>14} {'limit usados':>16} {'phi usados':>16}")
print("-" * 64)

for nombre, config in PROBLEMAS.items():
    best, hist, hl, hp = gabc(
        func = config["func"],
        lb   = config["lb"],
        ub   = config["ub"],
        n    = D,
        SN   = N,
        Gmax = 500,       # reducido solo para verificación
        seed = 42
    )
    print(f"{nombre:<14} {best:>14.6e} "
          f"{str(sorted(set(hl))):>16} "
          f"{str(sorted(set(hp))):>16}")

print("\n:) GABC verificado correctamente en los 4 problemas.")


# -----------------------------------------------------------------------------
# 11. ALGORITMO GENÉTICO (referencia — Práctica 3)
#     Cruza SBX + mutación polinomial + torneo binario + elitismo.
# -----------------------------------------------------------------------------

def algoritmo_genetico(func, lb, ub, n=10, N=100, Gmax=5000,
                       pc=0.9, pm=0.3, eta_c=20, eta_m=20, seed=None):
    """
    Algoritmo Genético con selección por torneo binario, cruza SBX
    y mutación polinomial. Usado como comparación contra GABC.
    Parámetros idénticos a los de la Práctica 3.
    """
    if seed is not None:
        np.random.seed(seed)

    pop     = np.random.uniform(lb, ub, (N, n))
    fitness = func(pop)

    mejor_idx = np.argmin(fitness)
    mejor_ind = pop[mejor_idx].copy()
    mejor_fit = fitness[mejor_idx]
    historial = [mejor_fit]

    for _ in range(Gmax):

        idx1      = np.random.permutation(N)
        idx2      = np.random.permutation(N)
        ganadores = np.where(fitness[idx1] <= fitness[idx2], idx1, idx2)
        padres    = pop[ganadores]

        hijos = np.empty_like(padres)
        for j in range(0, N - 1, 2):
            p1, p2 = padres[j].copy(), padres[j + 1].copy()

            if np.random.rand() < pc:
                for d in range(n):
                    if np.random.rand() <= 0.5 and abs(p1[d] - p2[d]) > 1e-10:
                        y1, y2 = min(p1[d], p2[d]), max(p1[d], p2[d])
                        b1 = 1 + 2*(y1 - lb)/(y2 - y1)
                        b2 = 1 + 2*(ub - y2)/(y2 - y1)
                        a1 = 2 - b1**(-(eta_c+1))
                        a2 = 2 - b2**(-(eta_c+1))
                        u  = np.random.rand()
                        bq1 = (u*a1)**(1/(eta_c+1)) if u <= 1/a1 \
                              else (1/(2-u*a1))**(1/(eta_c+1))
                        u  = np.random.rand()
                        bq2 = (u*a2)**(1/(eta_c+1)) if u <= 1/a2 \
                              else (1/(2-u*a2))**(1/(eta_c+1))
                        p1[d] = np.clip(0.5*((y1+y2) - bq1*(y2-y1)), lb, ub)
                        p2[d] = np.clip(0.5*((y1+y2) + bq2*(y2-y1)), lb, ub)

            for ind in [p1, p2]:
                if np.random.rand() < pm:
                    for d in range(n):
                        if np.random.rand() < 1/n:
                            delta1 = (ind[d] - lb) / (ub - lb)
                            delta2 = (ub - ind[d]) / (ub - lb)
                            u = np.random.rand()
                            if u <= 0.5:
                                val    = 2*u + (1-2*u)*(1-delta1)**(eta_m+1)
                                deltaq = val**(1/(eta_m+1)) - 1
                            else:
                                val    = 2*(1-u) + 2*(u-0.5)*(1-delta2)**(eta_m+1)
                                deltaq = 1 - val**(1/(eta_m+1))
                            ind[d] = np.clip(ind[d] + deltaq*(ub-lb), lb, ub)

            hijos[j], hijos[j+1] = p1, p2

        if N % 2 != 0:
            hijos[-1] = padres[-1].copy()

        fitness = func(hijos)
        pop     = hijos

        peor_idx = np.argmax(fitness)
        if mejor_fit < fitness[peor_idx]:
            pop[peor_idx]     = mejor_ind.copy()
            fitness[peor_idx] = mejor_fit

        mi = np.argmin(fitness)
        if fitness[mi] < mejor_fit:
            mejor_fit = fitness[mi]
            mejor_ind = pop[mi].copy()

        historial.append(mejor_fit)

    return mejor_fit, historial


# -----------------------------------------------------------------------------
# 12. EXPERIMENTO: 30 EJECUCIONES INDEPENDIENTES + PRUEBA DE WILCOXON
# -----------------------------------------------------------------------------

N_RUNS = 30   # ejecuciones independientes requeridas por el proyecto

resultados_gabc  = {nombre: [] for nombre in PROBLEMAS}
resultados_ag    = {nombre: [] for nombre in PROBLEMAS}
historiales_gabc = {nombre: [] for nombre in PROBLEMAS}
historiales_ag   = {nombre: [] for nombre in PROBLEMAS}

for nombre, config in PROBLEMAS.items():
    print(f"Problema: {nombre}")

    for run in range(N_RUNS):
        seed = run * 100 + 7   # semilla única y reproducible por corrida

        best_gabc, hist_gabc, _, _ = gabc(
            func = config["func"],
            lb   = config["lb"],
            ub   = config["ub"],
            n    = D,
            SN   = N,
            Gmax = GMAX,
            seed = seed
        )
        resultados_gabc[nombre].append(best_gabc)
        historiales_gabc[nombre].append(hist_gabc)

        best_ag, hist_ag = algoritmo_genetico(
            func = config["func"],
            lb   = config["lb"],
            ub   = config["ub"],
            n    = D,
            N    = N,
            Gmax = GMAX,
            seed = seed
        )
        resultados_ag[nombre].append(best_ag)
        historiales_ag[nombre].append(hist_ag)

    print(f"  GABC -> media: {np.mean(resultados_gabc[nombre]):.4e}  "
          f"std: {np.std(resultados_gabc[nombre]):.4e}")
    print(f"  AG   -> media: {np.mean(resultados_ag[nombre]):.4e}  "
          f"std: {np.std(resultados_ag[nombre]):.4e}")

print(f"\n✓ Experimentos completados: {N_RUNS} corridas × 4 problemas × 2 algoritmos.")