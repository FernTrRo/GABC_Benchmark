# GABC — Guided Artificial Bee Colony con Auto-adaptación de Parámetros

**Proyecto Final · Algoritmos Bioinspirados**  
Escuela Superior de Cómputo (ESCOM-IPN) — Ingeniería en Inteligencia Artificial  
Profesora: Dra. Miriam Pescador Rojas
Alumnos: 
1. Dustin Aburto González
2. Márquez Rosas José Humberto
3. Trujillo Rodríguez Fernanda

---

## Descripción

Implementación del algoritmo **GABC (Guided Artificial Bee Colony)** propuesto por Zhu & Kwong (2010), aplicado a cuatro funciones benchmark de optimización continua. El algoritmo incorpora:

- **Ecuación de perturbación guiada** que usa el mejor global (*gbest*) para acelerar la convergencia.
- **Auto-adaptación de parámetros en línea**: `limit` y `phi_scale` se ajustan dinámicamente cada 50 ciclos según la tasa de éxito de la población.
- **Fases vectorizadas** con NumPy (empleadas y observadoras sin bucles de Python), lo que reduce el tiempo de ejecución de minutos a segundos por corrida.

Los resultados se comparan contra un **Algoritmo Genético** (AG) con cruza SBX y mutación polinomial, usando 30 ejecuciones independientes y la prueba estadística de Wilcoxon.

---

## Funciones Benchmark

| Función     | Dominio             | Óptimo global |
|-------------|---------------------|---------------|
| Ackley      | [−32.768, 32.768]¹⁰ | f(0,...,0) = 0 |
| Griewank    | [−600, 600]¹⁰       | f(0,...,0) = 0 |
| Rastrigin   | [−5.12, 5.12]¹⁰     | f(0,...,0) = 0 |
| Rosenbrock  | [−2.048, 2.048]¹⁰   | f(1,...,1) = 0 |

Todas con **D = 10 dimensiones**, colonia de **N = 100** abejas y **5 000 ciclos** máximos.

---

## Estructura del Repositorio

```
.
├── gabc_benchmark.py       # Código fuente principal
├── GABC_corregido.ipynb    # Notebook con explicaciones y visualizaciones
└── README.md
```

---

## Requisitos

- Python 3.8 o superior
- NumPy
- SciPy
- Matplotlib

### Instalación de dependencias

```bash
pip install numpy scipy matplotlib
```

O con el archivo de entorno:

```bash
pip install -r requirements.txt
```

**`requirements.txt`** (crear manualmente si se desea):
```
numpy>=1.21
scipy>=1.7
matplotlib>=3.4
```

---

## Ejecución

### Ejecutar el script completo

```bash
python gabc_benchmark.py
```

El script realiza en orden:

1. **Verificación de óptimos**: comprueba que cada función devuelve f ≈ 0 en su óptimo conocido.
2. **Verificación rápida del GABC**: 500 ciclos en los 4 problemas con semilla fija.
3. **Experimento completo**: 30 corridas independientes de GABC y AG en los 4 problemas.

La salida esperada termina con:

```
✓ Experimentos completados: 30 corridas × 4 problemas × 2 algoritmos.
```

### Ejecutar sólo el notebook

```bash
jupyter notebook GABC_corregido.ipynb
```

El notebook incluye todas las explicaciones teóricas, pseudocódigos y gráficas de convergencia.

---

## Parámetros del Algoritmo

| Parámetro   | Valor inicial | Opciones de adaptación | Descripción                          |
|-------------|:-------------:|:----------------------:|--------------------------------------|
| `limit`     | 25            | {10, 25, 50}           | Ciclos sin mejora antes de abandonar |
| `phi_scale` | 0.8           | {0.5, 0.8, 1.0}        | Escala del factor de perturbación    |
| `psi`       | U(0, 1.5)     | —                       | Factor de guía hacia *gbest*         |
| `SN`        | 100           | —                       | Tamaño de la colonia                 |
| `Gmax`      | 5 000         | —                       | Máximo de ciclos                     |

**Regla de auto-adaptación** (cada 50 ciclos):

| Tasa de éxito | `limit` | `phi_scale` | Modo        |
|:---:|:---:|:---:|:---:|
| > 40 %  | 10 | 1.0 | Explotación agresiva |
| 10–40 % | 25 | 0.8 | Balance              |
| < 10 %  | 50 | 0.5 | Exploración          |

---

## Ecuación Principal (GABC)

La perturbación de cada dimensión `j` de la abeja `i` sigue:

```
v_ij = x_ij + phi_ij * (x_ij - x_kj) + psi_ij * (gbest_j - x_ij)
```

donde `k ≠ i` es un vecino aleatorio, `phi ~ U(-phi_scale, phi_scale)` y `psi ~ U(0, 1.5)`.

---

## Referencia

Zhu, G., & Kwong, S. (2010). *Gbest-guided artificial bee colony algorithm for numerical function optimization*. Applied Mathematics and Computation, 217(7), 3166–3173.

---
