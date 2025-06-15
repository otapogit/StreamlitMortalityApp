# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import re
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import requests
import json

st.set_page_config(layout="wide", page_title="Análisis de Mortalidad")

# Función para cargar y procesar datos
@st.cache_data
def cargar_datos():
    """Carga y procesa el dataset de mortalidad"""
    try:
        df = pd.read_csv('Defuncionesporedadsexo.csv', delimiter=";", encoding='utf-8', decimal=",")
        # Limpiar columna Total (quitar puntos y convertir a int)
        df["Total"] = df["Total"].astype(str).str.replace(".", "").astype(int)
        # Extraer código de causa de muerte
        df["Codigo_Causa"] = df["Causa de muerte"].apply(lambda x: re.split(r"\ +",x)[1].split(".")[0])
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        st.stop()

# Función para cargar datos geoespaciales de España
@st.cache_data
def cargar_geodatos():
    """Carga los datos geoespaciales de las comunidades autónomas de España"""
    try:
        # Probar diferentes fuentes de datos geoespaciales
        urls = [
            "https://raw.githubusercontent.com/deldersveld/topojson/master/countries/spain/spain-comunidad-with-canary-islands.json",
            "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/spain-communities.geojson"
        ]
        
        gdf = None
        for url in urls:
            try:
                gdf = gpd.read_file(url)
                break
            except:
                continue
        
        if gdf is None:
            raise Exception("No se pudo cargar ningún archivo geoespacial")
        
        # Mapeo actualizado para coincidir exactamente con los nombres de tu dataset
        mapeo_nombres = {
            'Andalusia': 'Andalucía',
            'Andalucía': 'Andalucía', 
            'Andalucia': 'Andalucía',
            'Aragon': 'Aragón', 
            'Aragón': 'Aragón',
            'Asturias': 'Asturias, Principado de',
            'Principado de Asturias': 'Asturias, Principado de',
            'Balearic Islands': 'Balears, Illes',
            'Islas Baleares': 'Balears, Illes',
            'Illes Balears': 'Balears, Illes',
            'Baleares': 'Balears, Illes',
            'Canary Islands': 'Canarias',
            'Canarias': 'Canarias',
            'Cantabria': 'Cantabria',
            'Castile and León': 'Castilla y León',
            'Castilla y León': 'Castilla y León',
            'Castilla-León': 'Castilla y León',
            'Castilla-Leon': 'Castilla y León',  # Añadida esta variante
            'Castile-La Mancha': 'Castilla - La Mancha',
            'Castilla-La Mancha': 'Castilla - La Mancha',
            'Catalonia': 'Cataluña',
            'Cataluña': 'Cataluña',
            'Catalunya': 'Cataluña',
            'Valencia': 'Comunitat Valenciana',
            'Comunidad Valenciana': 'Comunitat Valenciana',
            'Comunitat Valenciana': 'Comunitat Valenciana',
            'Extremadura': 'Extremadura',
            'Galicia': 'Galicia',
            'Madrid': 'Madrid, Comunidad de',
            'Comunidad de Madrid': 'Madrid, Comunidad de',
            'Murcia': 'Murcia, Región de',
            'Región de Murcia': 'Murcia, Región de',
            'Navarre': 'Navarra, Comunidad Foral de',
            'Navarra': 'Navarra, Comunidad Foral de',
            'Comunidad Foral de Navarra': 'Navarra, Comunidad Foral de',
            'Basque Country': 'País Vasco',
            'País Vasco': 'País Vasco',
            'Euskadi': 'País Vasco',
            'Pais Vasco': 'País Vasco',
            'La Rioja': 'Rioja, La',
            'Rioja': 'Rioja, La'
        }
        # Buscar la columna correcta con nombres
        nombre_columna = None
        for col in gdf.columns:
            if col.lower() in ['name', 'nombre', 'nam_ccaa', 'ccaa_name']:
                nombre_columna = col
                break
        
        if nombre_columna:
            gdf['Comunidad'] = gdf[nombre_columna].map(mapeo_nombres)
            # Si no se mapea, usar el nombre original
            gdf['Comunidad'] = gdf['Comunidad'].fillna(gdf[nombre_columna])
        else:
            raise Exception("No se encontró columna de nombres en el GeoDataFrame")
        
        return gdf
    except Exception as e:
        st.error(f"Error al cargar datos geoespaciales: {e}")
        # Crear un GeoDataFrame vacío como fallback
        return gpd.GeoDataFrame()

# Función para limpiar nombres de comunidades del dataset
def limpiar_nombre_comunidad(nombre):
    """Limpia el nombre de la comunidad quitando el número inicial"""
    if pd.isna(nombre):
        return nombre
    # Quitar números iniciales y espacios
    nombre_limpio = re.sub(r'^\d+\s+', '', str(nombre)).strip()
    return nombre_limpio

# Diccionario para convertir edades a valores numéricos
dic_edades = {
    "Menos de 1 año": 0.5, "De 1 a 14 años": 7.5, "De 15 a 29 años": 22,
    "De 30 a 39 años": 34.5, "De 40 a 44 años": 42, "De 45 a 49 años": 47,
    "De 50 a 54 años": 52, "De 55 a 59 años": 57, "De 60 a 64 años": 62,
    "De 65 a 69 años": 67, "De 70 a 74 años": 72, "De 75 a 79 años": 77,
    "De 80 a 84 años": 82, "De 85 a 89 años": 87, "De 90 a 94 años": 92,
    "95 y más años": 97
}

# Cargar datos
df = cargar_datos()
gdf_espana = cargar_geodatos()

# Preparar datos filtrados
df_todas_causas = df[df["Codigo_Causa"] == "I-XXII"]
df_causas_especificas = df[df["Codigo_Causa"] != "I-XXII"]

# Sidebar para navegación
st.sidebar.title("🏥 Análisis de Mortalidad")
pagina = st.sidebar.radio("Selecciona una página:", ["📊 Visualizaciones", "🗺️ Mapa por Comunidades"])

if pagina == "📊 Visualizaciones":
    st.title("📊 Análisis de Mortalidad - Visualizaciones")
    
    # CAMBIO: Cambiar de columnas a filas
    st.subheader("Evolución Temporal Total")
    
    @st.cache_data
    def datosmortalidad():    
        # Filtrar datos para mortalidad total por año
        df_evol = df_todas_causas[df_todas_causas["Edad"] == "Todas las edades" ]
        df_evol = df_evol[df_evol["Sexo"] == "Total"]
        df_evol = df_evol[df_evol["Total Nacional"] == "Total Nacional"]
        df_evol = df_evol.fillna("TODOS")
        df_evol = df_evol[df_evol["Comunidades y Ciudades Autónomas"] == "TODOS"]

        df_agrupado = df_evol.groupby("Periodo")["Total"].sum().reset_index()
        return df_agrupado
    
    df_agrupado = datosmortalidad()
    
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    ax1.plot(df_agrupado["Periodo"], df_agrupado["Total"], 
            marker='o', linewidth=2, markersize=5, color='steelblue')
    ax1.set_title("Evolución de Mortalidad Total por Año")
    ax1.set_xlabel("Año")
    ax1.set_ylabel("Total de Defunciones")
    ax1.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)
    
    # Separador
    st.markdown("---")
    
    # GRÁFICA 2: Comparación por causas principales
    st.subheader("Principales Causas de Muerte")
    
    # Obtener las principales causas
    causas_principales = ['II', 'IX', 'X', 'XI', 'XX']  # Tumores, Circulatorio, Respiratorio, Digestivo, Externas
    causas_seleccionadas = st.multiselect(
        "Selecciona causas:", 
        causas_principales,
        default=causas_principales[:3]
    )
    
    if causas_seleccionadas:
        
        @st.cache_data
        def datoscausas(causas):
            dfcausas = df_causas_especificas[df_causas_especificas["Edad"] == "Todas las edades"]
            listdfs = {}
            i=0
            for causa in causas:
                listdfs[i] = dfcausas[dfcausas["Codigo_Causa"] == causa]
                i+=1
            dfcausas = pd.concat(listdfs)
            dfcausas = dfcausas[dfcausas["Sexo"] == "Total"]
            dfcausas = dfcausas.fillna("TODOS")
            dfcausas = dfcausas[dfcausas["Comunidades y Ciudades Autónomas"] == "TODOS"]
            dfcausas = dfcausas[dfcausas["Total Nacional"] == "Total Nacional"]
                       
            return dfcausas
        df_causas = datoscausas(causas_seleccionadas)
        
        df_causas_agrup = df_causas.groupby(["Codigo_Causa", "Periodo"])["Total"].sum().reset_index()
        
        fig2, ax2 = plt.subplots(figsize=(15, 6))
        for causa in causas_seleccionadas:
            data_causa = df_causas_agrup[df_causas_agrup["Codigo_Causa"] == causa]
            ax2.plot(data_causa["Periodo"], data_causa["Total"], 
                    marker='o', label=f"Código {causa}", linewidth=2)
        
        ax2.set_title("Evolución por Tipo de Causa")
        ax2.set_xlabel("Año")
        ax2.set_ylabel("Total de Defunciones")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
    
    # Separador
    st.markdown("---")
    
    # GRÁFICA 3: Distribución por grupos de edad
    st.subheader("Distribución por Edad")
    
    # Selector de causa para análisis por edad
    causa_edad = st.selectbox(
        "Selecciona causa:", 
        list(df_causas_especificas["Codigo_Causa"].unique())
    )
    
    df_edad = df_causas_especificas[df_causas_especificas["Codigo_Causa"] == causa_edad]

    def causaedad():
        df_ed = df_edad[df_edad["Sexo"] == "Total"]
        df_ed = df_ed.fillna("TODOS")
        df_ed = df_ed[df_ed["Comunidades y Ciudades Autónomas"] == "TODOS"]
        df_ed = df_ed[df_ed["Total Nacional"] == "Total Nacional"]
        df_ed = df_ed[df_ed["Edad"] != "Todas las edades"]
        
        df_edad_agrup = df_ed.groupby("Edad")["Total"].sum().reset_index()
        df_edad_agrup["Edad_num"] = df_edad_agrup["Edad"].map(dic_edades)
        df_edad_agrup = df_edad_agrup.dropna().sort_values("Edad_num")
        
        return df_edad_agrup

    df_edad_agrup = causaedad()
    
    fig3, ax3 = plt.subplots(figsize=(15, 6))
    bars = ax3.bar(range(len(df_edad_agrup)), df_edad_agrup["Total"], 
                  color='orange', alpha=0.7)
    ax3.set_title(f"Mortalidad por Grupo de Edad - {causa_edad}")
    ax3.set_xlabel("Grupo de Edad")
    ax3.set_ylabel("Total de Defunciones")
    ax3.set_xticks(range(len(df_edad_agrup)))
    ax3.set_xticklabels(df_edad_agrup["Edad"], rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig3)

    # Separador
    st.markdown("---")
    
    # GRÁFICA 4: Comparación de Distribución por Edad entre Causas
    st.subheader("Comparación de Distribución de Edades por Causas")
    
    # Multiselector de causas para análisis por edad
    causas_disponibles = list(df_causas_especificas["Codigo_Causa"].unique())
    causas_boxplot = st.multiselect(
        "Selecciona las causas a comparar:", 
        causas_disponibles,
        default=causas_disponibles[:4] if len(causas_disponibles) >= 4 else causas_disponibles
    )

    if causas_boxplot:
        @st.cache_data
        def enfermedad_multi(causas_seleccionadas):
            # Filtrar por todas las causas seleccionadas
            df_edad_enfermedad = df_causas_especificas[df_causas_especificas["Codigo_Causa"].isin(causas_seleccionadas)]
            df_edad_enfermedad = df_edad_enfermedad[df_edad_enfermedad["Sexo"] == "Total"]
            df_edad_enfermedad = df_edad_enfermedad[df_edad_enfermedad["Total Nacional"] == "Total Nacional"]
            df_edad_enfermedad = df_edad_enfermedad.fillna("TODOS")
            df_edad_enfermedad = df_edad_enfermedad[df_edad_enfermedad["Comunidades y Ciudades Autónomas"] == "TODOS"]
            df_edad_enfermedad = df_edad_enfermedad[df_edad_enfermedad["Edad"] != "Todas las edades"]
            df_edad_enfermedad = df_edad_enfermedad.groupby(["Codigo_Causa", "Edad"], sort=False)["Total"].sum().reset_index()
            df_edad_enfermedad["Total"] = df_edad_enfermedad["Total"].apply(lambda x: int(x/10))
            df_edad_enfermedad["Edad"] = df_edad_enfermedad["Edad"].apply(lambda x: int(dic_edades[x]))
            
            return df_edad_enfermedad.loc[df_edad_enfermedad.index.repeat(df_edad_enfermedad['Total'])]
        
        expanded_data = enfermedad_multi(causas_boxplot)
        
        # Agrupar los valores de edad según la causa de muerte
        grouped_data = []
        labels = []
        for causa in causas_boxplot:
            causa_data = expanded_data[expanded_data["Codigo_Causa"] == causa]
            if not causa_data.empty:
                grouped_data.append(causa_data["Edad"].values)
                labels.append(causa)
        
        if grouped_data:
            # Crear el boxplot
            figg, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 6))
            # Crear el boxplot con colores
            box = ax.boxplot(grouped_data, labels=labels, patch_artist=True, 
                           medianprops=dict(color="red", linewidth=2), showfliers=False)
            
            colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#D4A5A5', '#FFB3E6', 
                     '#C2F0C2', '#FFD700', '#87CEEB', '#DDA0DD', '#F0E68C', '#98FB98',
                     '#F4A460', '#87CEFA', '#DEB887']
            
            # Aplicar colores a las cajas
            for i, (patch, color) in enumerate(zip(box['boxes'], colors)):
                patch.set(facecolor=color, alpha=0.7, edgecolor="black")
            
            # Personalizar bordes de bigotes
            for whisker in box['whiskers']:
                whisker.set(color="black", linewidth=1.2, linestyle="--")
            
            # Personalizar los valores atípicos
            for flier in box['fliers']:
                flier.set(marker="o", color="black", alpha=0.5)
            
            # Etiquetas
            ax.set_xlabel("Causa de Muerte", fontsize=12)
            ax.set_ylabel("Edad", fontsize=12)
            ax.set_title("Comparación de Distribución de Edades por Causa de Muerte", fontsize=14, fontweight="bold")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(figg)
            
            # Agregar estadísticas descriptivas
            st.subheader("Estadísticas por Causa")
            stats_data = []
            for causa in causas_boxplot:
                causa_data = expanded_data[expanded_data["Codigo_Causa"] == causa]
                if not causa_data.empty:
                    edades = causa_data["Edad"].values
                    stats_data.append({
                        "Causa": causa,
                        "Media": np.mean(edades),
                        "Mediana": np.median(edades),
                        "Desv. Estándar": np.std(edades),
                        "Min": np.min(edades),
                        "Max": np.max(edades)
                    })
            
            if stats_data:
                df_stats = pd.DataFrame(stats_data)
                df_stats = df_stats.round(2)
                st.dataframe(df_stats, use_container_width=True)
        else:
            st.warning("No hay datos disponibles para las causas seleccionadas.")
    else:
        st.info("Selecciona al menos una causa para mostrar el boxplot.")





elif pagina == "🗺️ Mapa por Comunidades":
    st.title("🗺️ Análisis por Comunidades Autónomas")
    
    # Controles del mapa
    col1, col2, col3 = st.columns(3)
    
    with col1:
        causa_mapa = st.selectbox(
            "Causa de muerte:", 
            ['I-XXII'] + list(df_causas_especificas["Codigo_Causa"].unique())
        )
    
    with col2:
        periodos_disponibles = sorted(df["Periodo"].unique(), reverse=True)
        periodo_mapa = st.selectbox("Período:", periodos_disponibles[:20])
    
    with col3:
        sexo_mapa = st.selectbox("Sexo:", ["Total", "Hombres", "Mujeres"])
    
    # Procesar datos para el mapa
    if causa_mapa == 'I-XXII':
        df_mapa_base = df_todas_causas
    else:
        df_mapa_base = df_causas_especificas[df_causas_especificas["Codigo_Causa"] == causa_mapa]
    
    # Filtrar datos y eliminar casos que agrupan todas las comunidades
    df_mapa = df_mapa_base[
        (df_mapa_base["Periodo"] == periodo_mapa) &
        (df_mapa_base["Edad"] == "Todas las edades") &
        (df_mapa_base["Sexo"] == sexo_mapa)
    ].copy()
    
    # Aplicar el mismo tratamiento que en el resto del código
    df_mapa = df_mapa.fillna("TODOS")
    df_mapa = df_mapa[df_mapa["Comunidades y Ciudades Autónomas"] != "TODOS"]
    
    if not df_mapa.empty and not gdf_espana.empty:
        # Limpiar nombres de comunidades usando la función específica
        df_mapa["Comunidad_limpia"] = df_mapa["Comunidades y Ciudades Autónomas"].apply(limpiar_nombre_comunidad)
        
        # Agrupar por comunidad y sumar totales
        df_mapa_agrup = df_mapa.groupby("Comunidad_limpia")["Total"].sum().reset_index()
        

        
        # Unir datos con geometrías
        gdf_mapa = gdf_espana.merge(df_mapa_agrup, left_on='Comunidad', right_on='Comunidad_limpia', how='left')
        gdf_mapa['Total'] = gdf_mapa['Total'].fillna(0)
        
        # Mostrar mapa con Folium
        st.subheader(f"Distribución Geográfica - {causa_mapa} ({periodo_mapa})")
        
        # Verificar si hay datos para mostrar
        datos_disponibles = gdf_mapa['Total'].sum() > 0
        if not datos_disponibles:
            st.warning("No hay datos disponibles para mostrar en el mapa con los filtros seleccionados.")
            st.write("**Datos disponibles por comunidad:**")
            st.write(df_mapa_agrup.sort_values('Total', ascending=False))
        
        # Crear mapa base centrado en España
        m = folium.Map(location=[40.0, -4.0], zoom_start=6)
        
        # Crear mapa coroplético si hay datos
        if datos_disponibles:
            # Normalizar valores para el color
            min_val = gdf_mapa[gdf_mapa['Total'] > 0]['Total'].min()
            max_val = gdf_mapa['Total'].max()
            
            @st.cache_data
            def get_color(value):
                if pd.isna(value) or value == 0:
                    return '#lightgray'
                # Normalizar entre 0 y 1
                normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
                # Escala de colores de amarillo a rojo
                if normalized < 0.25:
                    return '#FFFF00'  # Amarillo
                elif normalized < 0.5:
                    return '#FFAA00'  # Naranja claro
                elif normalized < 0.75:
                    return '#FF5500'  # Naranja
                else:
                    return '#FF0000'  # Rojo
            
            # Añadir cada comunidad como GeoJson
            for idx, row in gdf_mapa.iterrows():
                if pd.notna(row['geometry']) and pd.notna(row['Comunidad']):
                    total_str = f"{int(row['Total']):,}" if row['Total'] > 0 else "Sin datos"
                    
                    folium.GeoJson(
                        row['geometry'],
                        style_function=lambda feature, total=row['Total']: {
                            'fillColor': get_color(total),
                            'color': 'black',
                            'weight': 1,
                            'fillOpacity': 0.7,
                        },
                        highlight_function=lambda feature: {
                            'fillColor': 'blue',
                            'color': 'black',
                            'weight': 3,
                            'fillOpacity': 0.9,
                        },
                        tooltip=folium.Tooltip(f"<b>{row['Comunidad']}</b><br>Defunciones: {total_str}")
                    ).add_to(m)
            
            # Crear leyenda como HTML personalizado en Streamlit
            with st.container():
                st.markdown("""
                <div style="background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px; margin: 10px 0;">
                <strong>Leyenda - Nivel de Defunciones:</strong><br>
                <span style="background-color: #FFFF00; padding: 2px 8px; margin: 2px;">●</span> Bajo<br>
                <span style="background-color: #FFAA00; padding: 2px 8px; margin: 2px;">●</span> Medio-Bajo<br>
                <span style="background-color: #FF5500; padding: 2px 8px; margin: 2px;">●</span> Medio-Alto<br>
                <span style="background-color: #FF0000; padding: 2px 8px; margin: 2px;">●</span> Alto
                </div>
                """, unsafe_allow_html=True)
        
        # Mostrar el mapa
        st_folium(m, width=700, height=500)
        
        # Tabla y gráfico resumen
        col_tabla, col_grafico = st.columns(2)
        
        with col_tabla:
            st.subheader("Ranking por Comunidad")
            tabla_ranking = df_mapa_agrup[['Comunidad_limpia', 'Total']].sort_values('Total', ascending=False)
            tabla_ranking.columns = ['Comunidad', 'Total']
            # Formatear números con separadores de miles
            tabla_ranking['Total'] = tabla_ranking['Total'].apply(lambda x: f"{x:,}")
            st.dataframe(tabla_ranking, use_container_width=True)
        
        with col_grafico:
            st.subheader("Gráfico de Barras")
            # Convertir de vuelta a números para el gráfico
            tabla_ranking_num = df_mapa_agrup[['Comunidad_limpia', 'Total']].sort_values('Total', ascending=False)
            tabla_ranking_top10 = tabla_ranking_num.head(10)
            
            if not tabla_ranking_top10.empty:
                fig_barras, ax_barras = plt.subplots(figsize=(10, 8))
                bars = ax_barras.barh(tabla_ranking_top10['Comunidad_limpia'], tabla_ranking_top10['Total'])
                ax_barras.set_title(f"Top 10 Comunidades - {causa_mapa}")
                ax_barras.set_xlabel("Total Defunciones")
                
                # Invertir el orden para que el mayor aparezca arriba
                ax_barras.invert_yaxis()
                
                # Añadir valores en las barras
                for i, (idx, row) in enumerate(tabla_ranking_top10.iterrows()):
                    ax_barras.text(row['Total'], i, f" {row['Total']:,}", 
                                 va='center', ha='left', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig_barras)
            else:
                st.write("No hay datos para mostrar el gráfico")
    else:
        if gdf_espana.empty:
            st.error("No se pudieron cargar los datos geoespaciales. Verifica tu conexión a internet.")
        else:
            st.warning("No hay datos disponibles para la selección actual.")

# Información en sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📋 Códigos de Causas Principales")
    st.markdown("""
    - **I-XXII**: Todas las causas
    - **I**: Enfermedades infecciosas y parasitarias
    - **II**: Tumores
    - **III**: Enfermedades de la sangre y de los órganos hematopoyéticos, y ciertos trastornos que afectan al mecanismo de la inmunidad
    - **IV**: Enfermedades endocrinas, nutricionales y metabólicas
    - **V**: Trastornos mentales y del comportamiento
    - **VI-VIII**: Enfermedades del sistema nervioso y de los órganos de los sentidos
    - **IX**: Enfermedades del sistema circulatorio
    - **X**: Enfermedades del sistema respiratorio
    - **XI**: Enfermedades del sistema digestivo
    - **XII**: Enfermedades de la piel y del tejido subcutáneo
    - **XIII**: Enfermedades del sistema osteomuscular y del tejido conjuntivo
    - **XIV**: Enfermedades del sistema genitourinario
    - **XV**: Embarazo, parto y puerperio
    - **XVI**: Afecciones originadas en el periodo perinatal
    - **XVII**: Malformaciones congénitas, deformidades y anomalías cromosómicas
    - **XVIII**: Síntomas, signos y hallazgos anormales clínicos y de laboratorio, no clasificados en otra parte
    - **XX**: Causas externas de morbilidad y mortalidad
    """)
    
    st.markdown("### 📊 Información del Dataset")
    if 'df' in locals():
        st.write(f"**Total filas**: {len(df):,}")
        st.write(f"**Período**: {df['Periodo'].min()} - {df['Periodo'].max()}")
        st.write(f"**Comunidades**: {len(df['Comunidades y Ciudades Autónomas'].unique())}")
        st.write(f"**Causas**: {len(df['Codigo_Causa'].unique())}")
