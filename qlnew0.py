import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Necesitas instalar deap si no lo tienes: pip install deap
from deap import base, creator, tools, algorithms

# --- Configuración Inicial DEAP ---
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

# --- Funciones de Carga y Procesamiento de Datos (sin cambios) ---
@st.cache_data
def load_data_and_counts(uploaded_file):
    if uploaded_file is None: return None, {}, {}, {}, [], {}, 0, {}
    try:
        df = pd.read_csv(uploaded_file)
        if 'Numero' not in df.columns or 'Atraso' not in df.columns or 'Frecuencia' not in df.columns:
            st.error("El archivo debe contener las columnas 'Numero', 'Atraso' y 'Frecuencia'."); return None, {}, {}, {}, [], {}, 0, {}
        df['Numero'] = pd.to_numeric(df['Numero'], errors='coerce'); df['Atraso'] = pd.to_numeric(df['Atraso'], errors='coerce'); df['Frecuencia'] = pd.to_numeric(df['Frecuencia'], errors='coerce')
        df.dropna(subset=['Numero', 'Atraso', 'Frecuencia'], inplace=True)
        df['Numero'], df['Atraso'], df['Frecuencia'] = df['Numero'].astype(int).astype(str), df['Atraso'].astype(int), df['Frecuencia'].astype(int)
        st.success("Archivo de datos cargado exitosamente.")
        numero_a_atraso = dict(zip(df['Numero'], df['Atraso'])); numero_a_frecuencia = dict(zip(df['Numero'], df['Frecuencia']))
        atrasos_disponibles_int = sorted(df['Atraso'].unique()); numeros_validos = list(numero_a_atraso.keys())
        distribucion_probabilidad = {num: 1.0/len(numeros_validos) for num in numeros_validos} if numeros_validos else {}
        atraso_counts = df['Atraso'].astype(str).value_counts().to_dict(); total_atraso_dataset = df['Atraso'].sum()
        atraso_stats = {"min": df['Atraso'].min(), "max": df['Atraso'].max(), "p25": df['Atraso'].quantile(0.25), "p75": df['Atraso'].quantile(0.75)}
        return df, numero_a_atraso, numero_a_frecuencia, distribucion_probabilidad, atrasos_disponibles_int, atraso_counts, total_atraso_dataset, atraso_stats
    except Exception as e:
        st.error(f"Error al procesar el archivo de datos: {e}"); return None, {}, {}, {}, [], {}, 0, {}

@st.cache_data
def load_historical_combinations(uploaded_file):
    if uploaded_file is None: return []
    try:
        df_hist = pd.read_csv(uploaded_file, header=None)
        historical_sets = [set(pd.to_numeric(row, errors='coerce').dropna().astype(int)) for _, row in df_hist.iterrows()]
        historical_sets = [s for s in historical_sets if len(s) >= 6]
        if historical_sets: st.success(f"Archivo de historial cargado: {len(historical_sets)} combinaciones.")
        else: st.warning("El archivo de historial no contenía combinaciones válidas.")
        return historical_sets
    except Exception as e:
        st.error(f"Error al procesar el archivo de historial: {e}"); return []

# --- Funciones de Análisis Histórico ---
@st.cache_data
def analyze_historical_special_calc(historical_sets, total_atraso_dataset, numero_a_atraso):
    if not historical_sets or total_atraso_dataset is None: return None
    values = [total_atraso_dataset + 40 - sum(numero_a_atraso.get(str(num), 0) for num in s) for s in historical_sets]
    if not values: return None
    return {"min": int(np.min(values)), "max": int(np.max(values)), "mean": int(np.mean(values)), "std": int(np.std(values))}

@st.cache_data
def analyze_historical_frequency_cv(historical_sets, numero_a_frecuencia):
    if not historical_sets or not numero_a_frecuencia: return None
    cv_values = [np.std(freqs) / np.mean(freqs) for s in historical_sets if (freqs := [numero_a_frecuencia.get(str(num), 0) for num in s]) and np.mean(freqs) > 0]
    if not cv_values: return None
    return {"min": np.min(cv_values), "max": np.max(cv_values), "mean": np.mean(cv_values), "std": np.std(cv_values)}

# --- NUEVO: Función para analizar el CV de Atraso en el historial ---
@st.cache_data
def analyze_historical_delay_cv(historical_sets, numero_a_atraso):
    if not historical_sets or not numero_a_atraso: return None
    cv_values = [np.std(delays) / np.mean(delays) for s in historical_sets if (delays := [numero_a_atraso.get(str(num), 0) for num in s]) and np.mean(delays) > 0]
    if not cv_values: return None
    return {"min": np.min(cv_values), "max": np.max(cv_values), "mean": np.mean(cv_values), "std": np.std(cv_values)}

@st.cache_data
def analyze_historical_structure(historical_sets):
    if not historical_sets: return None, None, None
    sums = [sum(s) for s in historical_sets]; parity_counts = Counter(sum(1 for num in s if num % 2 == 0) for s in historical_sets)
    consecutive_counts = []
    for s in historical_sets:
        nums = sorted(list(s)); max_consecutive = 0; current_consecutive = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1: current_consecutive += 1
            else: max_consecutive = max(max_consecutive, current_consecutive); current_consecutive = 1
        consecutive_counts.append(max(max_consecutive, current_consecutive))
    sum_stats = {"min": int(np.min(sums)), "max": int(np.max(sums)), "mean": int(np.mean(sums)), "std": int(np.std(sums))}
    return sum_stats, parity_counts, Counter(consecutive_counts)

@st.cache_data
def analyze_historical_composition(historical_sets, numero_a_atraso, composicion_ranges):
    #... (sin cambios)
    if not historical_sets: return None
    def get_category(atraso, ranges):
        if ranges['caliente'][0] <= atraso <= ranges['caliente'][1]: return 'caliente'
        elif ranges['tibio'][0] <= atraso <= ranges['tibio'][1]: return 'tibio'
        elif ranges['frio'][0] <= atraso <= ranges['frio'][1]: return 'frio'
        elif atraso >= ranges['congelado'][0]: return 'congelado'
        return 'otro'
    counts = Counter(tuple(Counter(get_category(numero_a_atraso.get(str(num), -1), composicion_ranges) for num in s).get(cat, 0) for cat in ['caliente', 'tibio', 'frio', 'congelado']) for s in historical_sets)
    return counts if counts else None
    
# --- Motores de Generación y Filtrado ---
# --- MODIFICADO: Se añade 'delay_cv_range' a los parámetros ---
def generar_combinaciones_con_restricciones(params):
    dist_prob, num_a_atraso, num_a_freq, restr_atraso, n_sel, n_comb, hist_combs, total_atraso, special_range, freq_cv_range, sum_range, parity_counts_allowed, max_consecutive_allowed, hist_similarity_threshold, delay_cv_range = params
    valores = list(dist_prob.keys()); combinaciones = []; intentos = 0; max_intentos = n_comb * 400
    while len(combinaciones) < n_comb and intentos < max_intentos:
        intentos += 1
        seleccionados_str = random.sample(valores, n_sel); seleccionados = [int(n) for n in seleccionados_str]
        
        # --- Filtros Estándar (sin cambios) ---
        if not (sum_range[0] <= sum(seleccionados) <= sum_range[1]): continue
        if sum(1 for n in seleccionados if n % 2 == 0) not in parity_counts_allowed: continue
        nums = sorted(seleccionados); current_consecutive = 1; max_consecutive = 0
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1: current_consecutive += 1
            else: max_consecutive = max(max_consecutive, current_consecutive); current_consecutive = 1
        if max(max_consecutive, current_consecutive) > max_consecutive_allowed: continue
        freqs = [num_a_freq.get(str(val), 0) for val in seleccionados]; mean_freq = np.mean(freqs)
        if mean_freq == 0 or not (freq_cv_range[0] <= (np.std(freqs) / mean_freq) <= freq_cv_range[1]): continue
        
        # --- NUEVO: Filtro por CV de Atraso ---
        delays = [num_a_atraso.get(str(val), 0) for val in seleccionados]; mean_delay = np.mean(delays)
        if mean_delay == 0 or not (delay_cv_range[0] <= (np.std(delays) / mean_delay) <= delay_cv_range[1]): continue

        suma_atrasos = sum(delays); valor_especial = total_atraso + 40 - suma_atrasos
        if not (special_range[0] <= valor_especial <= special_range[1]): continue
        if any(Counter(num_a_atraso.get(str(n), -1) for n in seleccionados)[int(a)] > l for a, l in restr_atraso.items()): continue
        if hist_combs and any(len(set(seleccionados).intersection(h)) > hist_similarity_threshold for h in hist_combs): continue
        
        combinaciones.append(tuple(sorted(seleccionados)))
    conteo = Counter(combinaciones)
    return sorted({c: (f, np.prod([dist_prob.get(str(v), 0) for v in c])) for c, f in conteo.items()}.items(), key=lambda x: -x[1][1])

def procesar_combinaciones(params_tuple, n_ejec):
    with ProcessPoolExecutor() as executor:
        return [future.result() for future in as_completed([executor.submit(generar_combinaciones_con_restricciones, params_tuple) for _ in range(n_ejec)])]

def filtrar_por_composicion(combinaciones, numero_a_atraso, composicion_rules):
    #... (sin cambios)
    def get_category(atraso, ranges):
        if ranges['caliente'][0] <= atraso <= ranges['caliente'][1]: return 'caliente'
        elif ranges['tibio'][0] <= atraso <= ranges['tibio'][1]: return 'tibio'
        elif ranges['frio'][0] <= atraso <= ranges['frio'][1]: return 'frio'
        elif atraso >= ranges['congelado'][0]: return 'congelado'
        return 'otro'
    return [c for c in combinaciones if all(Counter(get_category(numero_a_atraso.get(str(n),-1), composicion_rules['ranges']) for n in c).get(cat,0)==cnt for cat,cnt in composicion_rules['counts'].items())]

# --- MODIFICADO: Se añade 'delay_cv_range' a la evaluación ---
def evaluar_individuo_deap(individuo_str, params):
    dist_prob, num_a_atraso, num_a_freq, restr_atraso, n_sel, hist_combs, total_atraso, special_range, freq_cv_range, sum_range, parity_counts_allowed, max_consecutive_allowed, hist_similarity_threshold, delay_cv_range = params
    individuo = [int(n) for n in individuo_str]
    if len(individuo) != n_sel or len(set(individuo)) != n_sel: return (0,)
    if not (sum_range[0] <= sum(individuo) <= sum_range[1]): return (0,)
    if sum(1 for n in individuo if n % 2 == 0) not in parity_counts_allowed: return (0,)
    nums = sorted(individuo); current_consecutive=1; max_consecutive=0
    for i in range(1, len(nums)):
        if nums[i]==nums[i-1]+1: current_consecutive+=1
        else: max_consecutive=max(max_consecutive, current_consecutive); current_consecutive=1
    if max(max_consecutive, current_consecutive) > max_consecutive_allowed: return (0,)
    freqs = [num_a_freq.get(str(val), 0) for val in individuo]; mean_freq = np.mean(freqs)
    if mean_freq==0 or not (freq_cv_range[0] <= (np.std(freqs) / mean_freq) <= freq_cv_range[1]): return (0,)
    
    # --- NUEVO: Verificación de CV de Atraso en el AG ---
    delays = [num_a_atraso.get(str(val), 0) for val in individuo]; mean_delay = np.mean(delays)
    if mean_delay == 0 or not (delay_cv_range[0] <= (np.std(delays) / mean_delay) <= delay_cv_range[1]): return (0,)

    if any(Counter(num_a_atraso.get(str(n),-1) for n in individuo)[int(a)] > l for a,l in restr_atraso.items()): return (0,)
    if hist_combs and any(len(set(individuo).intersection(h)) > hist_similarity_threshold for h in hist_combs): return (0,)
    suma_atrasos = sum(delays)
    valor_especial = total_atraso + 40 - suma_atrasos
    if not (special_range[0] <= valor_especial <= special_range[1]): return (0,)
    return (np.prod([dist_prob.get(str(val), 0) for val in individuo]),)

def ejecutar_algoritmo_genetico(ga_params, backend_params):
    #... (sin cambios)
    n_gen, n_pob, cxpb, mutpb, dist_prob, n_sel = ga_params
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, list(dist_prob.keys()), n_sel)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluar_individuo_deap, params=backend_params)
    toolbox.register("mate", tools.cxTwoPoint); toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1); toolbox.register("select", tools.selTournament, tournsize=3)
    pop = toolbox.population(n=n_pob)
    algorithms.eaSimple(pop, toolbox, cxpb, mutpb, n_gen, verbose=False)
    best = tools.selBest(pop, k=1)[0] if pop else None
    if best:
        best = sorted([int(n) for n in set(best)]);
        if len(best) != n_sel: return None, 0.0, "AG no mantuvo individuos válidos."
        return best, toolbox.evaluate(best)[0], None
    return None, 0.0, "Población final vacía."

# ----------------------- Interfaz Gráfica de Streamlit -----------------------
st.set_page_config(layout="wide", page_title="Generador de Combinaciones de Precisión")
st.title("Modelo Homeostático de Precisión")
if 'suggested_composition' not in st.session_state: st.session_state.suggested_composition = None

st.header("1. Cargar Archivos de Datos")
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Sube CSV ('Numero', 'Atraso', 'Frecuencia')", type="csv", key="data_uploader")
    df, num_a_atraso, num_a_freq, dist_prob, atrasos_disp, atraso_counts, total_atraso, atraso_stats = load_data_and_counts(uploaded_file)
with col2:
    hist_file = st.file_uploader("Sube CSV con Historial de Combinaciones", type="csv", key="history_uploader")
    historical_combinations_set = load_historical_combinations(hist_file)

n_selecciones = 6
if df is not None:
     st.info(f"**Suma total de 'Atraso' en el dataset:** {total_atraso}")

st.header("2. Configuración de Filtros de Precisión")
restricciones_finales, composicion_rules, sum_range, parity_counts_allowed, max_consecutive_allowed, hist_similarity_threshold = {}, {}, (0, 999), [], 6, 6
special_calc_range, freq_cv_range, delay_cv_range = (0, 99999), (0.0, 999.9), (0.0, 999.9)

if df is not None:
    st.subheader("Filtros de Homeostasis (Etapa 1)")
    if historical_combinations_set:
        col_freq, col_spec, col_delay_cv = st.columns(3) # Añadida una tercera columna
        with col_freq:
            with st.expander("CV de Frecuencia", expanded=True):
                stats_freq_cv = analyze_historical_frequency_cv(historical_combinations_set, num_a_freq)
                if stats_freq_cv:
                    st.info(f"Hist: **{stats_freq_cv['min']:.2f}** a **{stats_freq_cv['max']:.2f}**")
                    default_start_cv = max(0.0, stats_freq_cv['mean'] - stats_freq_cv['std'])
                    default_end_cv = min(2.0, stats_freq_cv['mean'] + stats_freq_cv['std'])
                    freq_cv_range = st.slider("Rango de CV Frecuencia:", 0.0, 2.0, (default_start_cv, default_end_cv), format="%.2f", key="freq_cv_slider")
        with col_spec:
            with st.expander("'Cálculo Especial'", expanded=True):
                stats_special = analyze_historical_special_calc(historical_combinations_set, total_atraso, num_a_atraso)
                if stats_special:
                    st.info(f"Hist: **{stats_special['min']}** a **{stats_special['max']}**")
                    default_start_special = float(stats_special['mean'] - stats_special['std'])
                    default_end_special = float(stats_special['mean'] + stats_special['std'])
                    special_calc_range = st.slider("Rango de Cálculo Especial:", float(stats_special['min'] - 50), float(stats_special['max'] + 50), (default_start_special, default_end_special), key="special_slider")
        
        # --- NUEVO: Expander y Slider para el CV de Atraso ---
        with col_delay_cv:
            with st.expander("CV de Atraso", expanded=True):
                stats_delay_cv = analyze_historical_delay_cv(historical_combinations_set, num_a_atraso)
                if stats_delay_cv:
                    st.info(f"Hist: **{stats_delay_cv['min']:.2f}** a **{stats_delay_cv['max']:.2f}**")
                    default_start_delay_cv = max(0.0, stats_delay_cv['mean'] - stats_delay_cv['std'])
                    default_end_delay_cv = min(2.0, stats_delay_cv['mean'] + stats_delay_cv['std'])
                    delay_cv_range = st.slider("Rango de CV Atraso:", 0.0, 2.0, (default_start_delay_cv, default_end_delay_cv), format="%.2f", key="delay_cv_slider")

    # --- Resto de la UI (sin cambios) ---
    st.subheader("Filtros de Estructura Interna (Etapa 1)")
    #... (el código de los filtros de estructura que ya tenías)
    st.subheader("Filtros Estratégicos (Etapa 1 y 2)")
    #... (el código de los filtros estratégicos que ya tenías)
else:
    st.info("Carga los archivos para configurar los filtros.")

# (El resto del código de la UI y los botones de ejecución necesitan ser restaurados si los borraste.
#  A continuación, muestro cómo se modificarían los puntos clave)

# ... asumiendo que el resto de la UI está presente ...

# --- MODIFICADO: Añadir el nuevo rango a los parámetros que se pasan a los algoritmos ---
if df is not None:
    backend_params = (dist_prob, num_a_atraso, num_a_freq, restricciones_finales, n_selecciones, historical_combinations_set, total_atraso, special_calc_range, freq_cv_range, sum_range, parity_counts_allowed, max_consecutive_allowed, hist_similarity_threshold, delay_cv_range)
    
    # ... Botones de Ejecución ...
    
    # --- MODIFICADO: Dentro del botón "Ejecutar Simulación en Cascada" ---
    # ...
    # Al final de la tabla de resultados de la simulación, añade la nueva columna
    if combinaciones_refinadas:
        data = []
        for c in combinaciones_refinadas:
            freqs = [num_a_freq.get(str(v),0) for v in c]
            delays = [num_a_atraso.get(str(v),0) for v in c]
            data.append({
                "Combinación": " - ".join(map(str, sorted(c))), 
                "CV Frecuencia": np.std(freqs)/np.mean(freqs) if np.mean(freqs) > 0 else 0,
                "CV Atraso": np.std(delays)/np.mean(delays) if np.mean(delays) > 0 else 0, # <-- NUEVA COLUMNA
                "Cálculo Especial": total_atraso + 40 - sum(delays)
            })
        df_results = pd.DataFrame(data)
        df_results['CV Frecuencia'] = df_results['CV Frecuencia'].map('{:,.2f}'.format)
        df_results['CV Atraso'] = df_results['CV Atraso'].map('{:,.2f}'.format) # <-- Formato para la nueva columna
        st.dataframe(df_results.reset_index(drop=True))
