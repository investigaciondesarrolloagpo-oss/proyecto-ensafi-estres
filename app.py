import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Estrés Financiero ENSAFI 2023", page_icon="📊", layout="wide")

@st.cache_data
def cargar_datos():
    df = pd.read_csv("datos_procesados.csv")
    df['Genero'] = pd.Categorical(df['Genero'], categories=['Hombre', 'Mujer'])
    df['Academia'] = pd.Categorical(df['Academia'], categories=['No Universitario', 'Universitario'])
    df['Vulnerabilidad'] = pd.Categorical(df['Vulnerabilidad'], categories=['No Vulnerable', 'Vulnerable No Pobre', 'Pobreza Alta'], ordered=True)
    df['Rango_Estres'] = pd.Categorical(df['Rango_Estres'], categories=['Sin estrés financiero', 'Estrés bajo', 'Estrés moderado', 'Estrés alto', 'Estrés muy alto / crítico'], ordered=True)
    return df

with st.spinner("⏳ Cargando datos..."):
    df_total = cargar_datos()

st.success(f"✅ Datos cargados: **{len(df_total):,}** registros.")

st.title("Efectos opuestos de la pobreza material y el género femenino")
st.header("sobre el estrés financiero en universitarios mexicanos")
st.markdown("**Proyecto final — Fundamentos de Análisis de Datos** | Fuente: ENSAFI 2023 | Metodología: Boltvinik adaptado | Modelo: OLS + ANOVA")

# TABLA 1
total_n = len(df_total)
tabla1_rows = [{'Grupo': 'Total Encuestados', 'n': total_n, 'Porcentaje': 100.0}]
for cat in df_total['Academia'].cat.categories:
    n = (df_total['Academia'] == cat).sum()
    tabla1_rows.append({'Grupo': cat, 'n': n, 'Porcentaje': round(n/total_n*100, 2)})
for cat in df_total['Genero'].cat.categories:
    n = (df_total['Genero'] == cat).sum()
    tabla1_rows.append({'Grupo': cat, 'n': n, 'Porcentaje': round(n/total_n*100, 2)})
for cat in df_total['Vulnerabilidad'].cat.categories:
    n = (df_total['Vulnerabilidad'] == cat).sum()
    tabla1_rows.append({'Grupo': cat, 'n': n, 'Porcentaje': round(n/total_n*100, 2)})
tabla1 = pd.DataFrame(tabla1_rows)

# TABLA 2
grupos_list = []
def add_group(subset, nombre):
    temp = subset[['Rango_Estres']].copy()
    temp['Grupo'] = nombre
    return temp

grupos_list.append(add_group(df_total, 'Total General'))
grupos_list.append(add_group(df_total[df_total['Academia'] == 'No Universitario'], 'No Universitario'))
grupos_list.append(add_group(df_total[df_total['Academia'] == 'Universitario'], 'Universitario'))
grupos_list.append(add_group(df_total[df_total['Genero'] == 'Hombre'], 'Hombre'))
grupos_list.append(add_group(df_total[df_total['Genero'] == 'Mujer'], 'Mujer'))
grupos_list.append(add_group(df_total[df_total['Vulnerabilidad'] == 'No Vulnerable'], 'No Vulnerable'))
grupos_list.append(add_group(df_total[df_total['Vulnerabilidad'] == 'Vulnerable No Pobre'], 'Vulnerable No Pobre'))
grupos_list.append(add_group(df_total[df_total['Vulnerabilidad'] == 'Pobreza Alta'], 'Pobreza Alta'))

df_grupos = pd.concat(grupos_list, ignore_index=True)
tabla2 = pd.crosstab(df_grupos['Grupo'], df_grupos['Rango_Estres'])
orden = ['Total General', 'No Universitario', 'Universitario', 'Hombre', 'Mujer', 'No Vulnerable', 'Vulnerable No Pobre', 'Pobreza Alta']
tabla2 = tabla2.reindex([o for o in orden if o in tabla2.index])

# TABLA 3: REGRESIÓN
df_reg = df_total.copy()
df_reg['Academia_Univ'] = (df_reg['Academia'] == 'Universitario').astype(int)
df_reg['Genero_Mujer'] = (df_reg['Genero'] == 'Mujer').astype(int)
df_reg['Vuln_PobrezaAlta'] = (df_reg['Vulnerabilidad'] == 'Pobreza Alta').astype(int)
df_reg['Vuln_Vulnerable'] = (df_reg['Vulnerabilidad'] == 'Vulnerable No Pobre').astype(int)

X = df_reg[['Academia_Univ', 'Genero_Mujer', 'Vuln_PobrezaAlta', 'Vuln_Vulnerable']]
X = sm.add_constant(X)
y = df_reg['score']
modelo = sm.OLS(y, X).fit()

resultados = pd.DataFrame({
    'term': ['(Intercept)', 'AcademiaUniversitario', 'GeneroMujer',
             'VulnerabilidadPobreza Alta', 'VulnerabilidadVulnerable No Pobre'],
    'estimate': modelo.params.values,
    'std.error': modelo.bse.values,
    'statistic': modelo.tvalues.values,
    'p.value': modelo.pvalues.values
})
resultados['Significancia'] = resultados['p.value'].apply(
    lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
)

# GRÁFICO
sig_factors = resultados[
    (resultados['term'] != '(Intercept)') & (resultados['Significancia'] != 'ns')
].copy()
sig_factors['Factor'] = sig_factors['term'].str.replace('Academia|Genero|Vulnerabilidad', '', regex=True)
colors = ['steelblue' if x < 0 else 'firebrick' for x in sig_factors['estimate']]

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(sig_factors['Factor'], sig_factors['estimate'], color=colors)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Coeficiente Beta', fontsize=12)
ax.set_ylabel('Variable', fontsize=12)
ax.set_title('Factores con Impacto Significativo en el Estrés Financiero', fontsize=14, fontweight='bold')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='steelblue', label='Reduce Estrés'), Patch(facecolor='firebrick', label='Aumenta Estrés')]
ax.legend(handles=legend_elements, title='Efecto', loc='lower right')
for i, (idx, row) in enumerate(sig_factors.iterrows()):
    ax.text(row['estimate'], i, f" {row['estimate']:.3f}",
            va='center', ha='left' if row['estimate'] > 0 else 'right',
            fontsize=10, fontweight='bold')
plt.tight_layout()

# TABLA 4
mask_sig = (
    (df_total['Genero'] == 'Mujer') |
    (df_total['Academia'] == 'Universitario') |
    (df_total['Vulnerabilidad'] == 'Pobreza Alta')
)
df_problemas = df_total[mask_sig].copy()

tabla4 = df_problemas.groupby(
    ['Vulnerabilidad', 'Academia', 'Genero'], observed=True
).agg(
    Casos=('score', 'count'),
    Estres_Promedio=('score', 'mean'),
    Pct_Sin_Ahorro=('Sin_Ahorro', lambda x: x.mean() * 100 if x.notna().any() else np.nan),
    Pct_Imprevistos=('Incapacidad_Imprevistos', lambda x: x.mean() * 100 if x.notna().any() else np.nan)
).reset_index()
tabla4 = tabla4[tabla4['Casos'] > 0].sort_values('Estres_Promedio', ascending=False)

# TABLA 5: ZOOMS
def safe_anova(df_subset, group_cols):
    if len(df_subset) == 0:
        return None, None
    grupos = [g['score'].dropna().values for _, g in df_subset.groupby(group_cols, observed=True)]
    grupos = [g for g in grupos if len(g) > 0]
    if len(grupos) < 2:
        return None, None
    return stats.f_oneway(*grupos)

mujer_df = df_total[df_total['Genero'] == 'Mujer']
mujer_zoom = mujer_df.groupby(['Academia', 'Vulnerabilidad'], observed=True).agg(
    n=('score', 'count'), Estres_Promedio=('score', 'mean'), Estres_Std=('score', 'std')
).reset_index()
f_m, p_m = safe_anova(mujer_df, ['Academia', 'Vulnerabilidad'])
tukey_m_txt = ""
if f_m is not None:
    mujer_df['grupo_tukey'] = mujer_df['Academia'].astype(str) + ' + ' + mujer_df['Vulnerabilidad'].astype(str)
    tukey_m = pairwise_tukeyhsd(mujer_df['score'], mujer_df['grupo_tukey'])
    tukey_m_txt = str(tukey_m)

univ_df = df_total[df_total['Academia'] == 'Universitario']
univ_zoom = univ_df.groupby(['Genero', 'Vulnerabilidad'], observed=True).agg(
    n=('score', 'count'), Estres_Promedio=('score', 'mean'), Estres_Std=('score', 'std')
).reset_index()
f_u, p_u = safe_anova(univ_df, ['Genero', 'Vulnerabilidad'])
tukey_u_txt = ""
if f_u is not None:
    univ_df['grupo_tukey'] = univ_df['Genero'].astype(str) + ' + ' + univ_df['Vulnerabilidad'].astype(str)
    tukey_u = pairwise_tukeyhsd(univ_df['score'], univ_df['grupo_tukey'])
    tukey_u_txt = str(tukey_u)

pob_df = df_total[df_total['Vulnerabilidad'] == 'Pobreza Alta']
pob_zoom = pob_df.groupby(['Genero', 'Academia'], observed=True).agg(
    n=('score', 'count'), Estres_Promedio=('score', 'mean'), Estres_Std=('score', 'std')
).reset_index()
f_p, p_p = safe_anova(pob_df, ['Genero', 'Academia'])
tukey_p_txt = ""
if f_p is not None:
    pob_df['grupo_tukey'] = pob_df['Genero'].astype(str) + ' + ' + pob_df['Academia'].astype(str)
    tukey_p = pairwise_tukeyhsd(pob_df['score'], pob_df['grupo_tukey'])
    tukey_p_txt = str(tukey_p)

# INTERFAZ
tab_intro, tab_t1, tab_t2, tab_t3, tab_t4, tab_t5 = st.tabs([
    "📝 Introducción", "📊 Tabla 1", "📈 Tabla 2", "🎯 Tabla 3", "📉 Gráfico & Tabla 4", "🔍 Tabla 5"
])

with tab_intro:
    st.subheader("Resumen del análisis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total encuestados", f"{len(df_total):,}")
    col2.metric("Universitarios", f"{(df_total['Academia']=='Universitario').sum():,}")
    col3.metric("R² del modelo", f"{modelo.rsquared:.4f}")
    st.markdown("Dashboard interactivo con resultados de la ENSAFI 2023. Metodología: Boltvinik adaptado, regresión OLS y ANOVA.")

with tab_t1:
    st.subheader("Tabla 1: Composición Sociodemográfica")
    st.dataframe(tabla1, use_container_width=True, hide_index=True)

with tab_t2:
    st.subheader("Tabla 2: Rangos de Estrés por Grupo")
    st.dataframe(tabla2, use_container_width=True)

with tab_t3:
    st.subheader("Tabla 3: Impacto Sociodemográfico en el Nivel de Estrés")
    st.dataframe(resultados, use_container_width=True, hide_index=True)
    st.metric("R² del modelo", f"{modelo.rsquared:.4f}")
    st.caption("*** p<0.001; ** p<0.01; * p<0.05. Base: No Universitario, Hombre, No Vulnerable.")

with tab_t4:
    col_g, col_p = st.columns([1, 1])
    with col_g:
        st.subheader("Gráfico: Coeficientes Significativos")
        st.pyplot(fig)
    with col_p:
        st.subheader("Tabla 4: Problemas Financieros Reales")
        st.dataframe(tabla4, use_container_width=True, hide_index=True)

with tab_t5:
    st.subheader("Tabla 5: Análisis Interseccional (Zooms)")
    
    st.markdown("**5.1 Mujer: Estrés por subgrupos**")
    st.dataframe(mujer_zoom, use_container_width=True, hide_index=True)
    if f_m:
        st.write(f"**ANOVA:** F={f_m:.3f}, p={p_m:.2e}")
        with st.expander("Ver Tukey HSD — Mujer"):
            st.text(tukey_m_txt)
    
    st.divider()
    st.markdown("**5.2 Universitario: Estrés por subgrupos**")
    st.dataframe(univ_zoom, use_container_width=True, hide_index=True)
    if f_u:
        st.write(f"**ANOVA:** F={f_u:.3f}, p={p_u:.2e}")
        with st.expander("Ver Tukey HSD — Universitario"):
            st.text(tukey_u_txt)
    
    st.divider()
    st.markdown("**5.3 Pobreza Alta: Estrés por subgrupos**")
    st.dataframe(pob_zoom, use_container_width=True, hide_index=True)
    if f_p:
        st.write(f"**ANOVA:** F={f_p:.3f}, p={p_p:.2e}")
        with st.expander("Ver Tukey HSD — Pobreza Alta"):
            st.text(tukey_p_txt)

st.sidebar.header("⚙️ Acerca del proyecto")
st.sidebar.info("""
- **Curso:** Fundamentos de Análisis de Datos  
- **Herramientas:** Python, Pandas, Statsmodels, Streamlit  
- **Datos:** ENSAFI 2023 (INEGI)
""")
st.sidebar.markdown("💻 [Ver código en GitHub](https://github.com/investigaciondesarrolloagpo-oss/proyecto-ensafi-estres)")
