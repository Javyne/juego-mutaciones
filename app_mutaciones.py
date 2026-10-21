"""Abrir con: python -m streamlit run app_mutaciones.py"""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from motor_mutaciones import (
    MODELOS, TOTAL_RONDAS, chi2_bondad_ajuste, evaluar_respuesta,
    generar_ronda, modelos_para_validar,
)


def iniciar_partida():
    """El estado conserva los datos al interactuar con los controles."""
    st.session_state.rng = np.random.default_rng()
    st.session_state.ronda = generar_ronda(st.session_state.rng)
    st.session_state.historial = []
    st.session_state.respuesta = None


def mostrar_histograma(datos):
    x = np.arange(int(datos.max()) + 1)
    frecuencias = 100 * np.bincount(datos, minlength=len(x)) / len(datos)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(x, frecuencias, color="#2980b9")
    ax.set(xlabel="Mutaciones por región", ylabel="Réplicas (%)",
           title="¿Qué modelo generó estos conteos?")
    ax.xaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def mostrar_validacion(ronda):
    """Los gráficos de resolución se muestran después de responder."""
    datos = ronda["datos"]
    x = np.arange(int(datos.max()) + 1)
    frecuencias = 100 * np.bincount(datos, minlength=len(x)) / len(datos)
    probabilidades = (np.arange(1, len(datos) + 1) - 0.5) / len(datos)
    for nombre, distribucion in modelos_para_validar(ronda):
        st.markdown(f"**Validación con {nombre}**")
        fig, ejes = plt.subplots(1, 2, figsize=(10, 3.5))
        ejes[0].bar(x, frecuencias, alpha=0.65, label="Observado")
        ejes[0].plot(x, 100 * distribucion.pmf(x), "o-",
                     color="#d35400", markersize=3, label=nombre)
        ejes[0].set(xlabel="Mutaciones por región", ylabel="Réplicas (%)",
                    title="Datos y probabilidades del modelo")
        ejes[0].legend()
        cuantiles = distribucion.ppf(probabilidades)
        tope = max(float(cuantiles.max()), float(datos.max()), 1)
        ejes[1].scatter(cuantiles, np.sort(datos), s=12, alpha=0.5)
        ejes[1].plot([0, tope], [0, tope], "r--")
        ejes[1].set(xlabel="Cuantiles del modelo", ylabel="Cuantiles observados",
                    title="QQ-plot: comparar con la recta roja")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        estadistico, gl, p = chi2_bondad_ajuste(datos, distribucion)
        if np.isnan(p):
            st.info("χ² no calculable: quedan muy pocas categorías agrupadas. "
                    "No permite aceptar ni rechazar el modelo.")
        else:
            st.write(f"χ² = {estadistico:.2f} · grados de libertad = {gl} · p = {p:.3g}")
            if p < 0.05:
                st.write("Con un umbral de 0,05, hay evidencia contra este ajuste.")
            else:
                st.write("Con un umbral de 0,05, no se rechaza este ajuste. "
                         "Esto no prueba que sea el modelo verdadero.")
        if nombre == "Binomial negativa":
            st.caption("Se usa el α real revelado y se estima solo la media, como en "
                       "el notebook. No es un ajuste completo de ambos parámetros.")
    st.caption("En el QQ-plot, puntos cerca de la recta indican concordancia. "
               "Los escalones son normales en conteos. El test χ² es aproximado; "
               "incluso el generador correcto puede ser rechazado por azar.")



def mostrar_interpretacion_biologica(ronda):
    """Traduce los números de esta ronda a su significado biológico."""
    datos = ronda["datos"]
    media_real = ronda["media"]
    longitud = len(ronda["secuencia"])
    media_observada = float(datos.mean())
    tasa_observada = media_observada / longitud
    porcentaje_ceros = 100 * float(np.mean(datos == 0))

    st.subheader("¿Qué significa biológicamente esta ronda?")
    st.markdown("**La media: cuántas mutaciones esperamos en esta región**")
    st.write(
        f"En estas copias de una región de **{longitud} nucleótidos**, observamos "
        f"un promedio de **{media_observada:.3f} mutaciones por copia** durante "
        f"el período estudiado. La simulación tenía una media de **{media_real:.3f}**. "
        f"Si observáramos otras 100 copias en las mismas condiciones, esperaríamos "
        f"aproximadamente **{100 * media_real:.1f} eventos de mutación en total**, "
        "aunque el número efectivo variaría por azar."
    )
    st.write(
        "Una media decimal no significa que exista una fracción de mutación en una "
        "copia: cada copia tiene un conteo entero. El promedio reúne copias con "
        "distintos conteos, incluidas las que no tuvieron ninguna mutación."
    )
    st.markdown("**La tasa por nucleótido: comparar regiones de distinto tamaño**")
    st.write(
        f"Al dividir el promedio observado por la longitud, estimamos "
        f"**{tasa_observada:.5f} eventos por nucleótido y por período** "
        f"(unos **{1000 * tasa_observada:.2f} por cada 1000 nucleótidos**). "
        "Una región más larga puede acumular más mutaciones simplemente porque "
        "tiene más sitios donde pueden ocurrir. Para comparar su propensión a "
        "mutar, hay que considerar tanto la longitud como el tiempo de observación."
    )
    st.caption(
        "El juego no define la duración del período: estos números no se pueden "
        "expresar por año ni por generación. Son parámetros didácticos, no una "
        "estimación de la tasa de una especie concreta. Contamos eventos; la tasa "
        "por sitio no equivale exactamente al porcentaje de sitios distintos mutados."
    )

    st.markdown("**El modelo: cómo se reparten las mutaciones entre las copias**")
    if ronda["modelo"] == 1:
        st.write(
            "**Poisson representa una propensión a mutar común a todas las copias**, "
            "con eventos independientes en el modelo. Algunas copias tienen más "
            "mutaciones que otras por azar, aun compartiendo esa propensión. "
            "Biológicamente, sería compatible con condiciones de exposición y "
            "mecanismos de replicación y reparación similares entre las copias. "
            "El ajuste estadístico no demuestra que esos mecanismos sean idénticos."
        )
    else:
        st.write(
            "**La binomial negativa representa copias con distinta propensión a mutar.** "
            "En la simulación algunas reciben tasas bajas y otras altas. Como "
            "hipótesis biológicas, podrían existir diferencias entre células en "
            "la exposición a agentes mutágenos o en la fidelidad de replicación "
            "y reparación del ADN. El juego simula esa heterogeneidad mediante "
            "tasas aleatorias; no simula ni identifica un mecanismo molecular concreto."
        )
        st.write(
            "Por eso una misma media puede ocultar una distribución desigual: "
            "muchas copias acumulan pocas mutaciones y algunas acumulan muchas. "
            "El parámetro α describe cuánta heterogeneidad se introdujo, no un "
            "porcentaje de ADN dañado. La sobredispersión por sí sola tampoco "
            "demuestra selección natural ni localiza sitios de alta mutación."
        )

    st.write(
        f"**En tus datos, el {porcentaje_ceros:.1f} % de las copias no presentó "
        "eventos de mutación durante el período simulado.** Esto se refiere solo "
        "a la región y al intervalo observados; no implica que todo el genoma "
        "de esas copias esté libre de cambios."
    )
    st.info(
        "**Conclusión biológica:** la media resume cuántos eventos se acumulan; "
        "el modelo describe cómo se distribuyen entre copias. Un mayor conteo "
        "no indica por sí solo mayor daño: las mutaciones pueden ser neutras, "
        "perjudiciales o beneficiosas según dónde ocurran y qué función alteren. "
        "Aquí no conocemos sus posiciones ni sus efectos, y la secuencia mostrada "
        "no interviene en la simulación; no podemos inferir enfermedad, adaptación "
        "ni consecuencias sobre una proteína."
    )
    with st.expander("Fuentes para la interpretación biológica"):
        st.markdown(
            "- [Heterogeneidad de tasas y distribución de los conteos "
            "(Alexander y colaboradores)]"
            "(https://pmc.ncbi.nlm.nih.gov/articles/PMC5850754/)\n"
            "- [Ejemplo de uso de Gamma-Poisson en conteos de mutaciones: NIMBus]"
            "(https://pmc.ncbi.nlm.nih.gov/articles/PMC7580035/)\n"
            "- [Qué es una mutación y qué efectos puede tener (NCI)]"
            "(https://www.cancer.gov/publications/dictionaries/genetics-dictionary/def/mutation)"
        )


def mostrar_resultado(ronda, respuesta):
    datos = ronda["datos"]
    media = float(datos.mean())
    varianza = float(datos.var(ddof=1))
    indice = varianza / media if media > 0 else float("nan")
    st.subheader("Lo que nos enseñan los resultados")
    st.success(f"Puntaje: {respuesta['puntos']}/100 · "
               f"Media: {respuesta['puntos_media']}/60 · "
               f"Modelo: {respuesta['puntos_modelo']}/40")
    acierto = respuesta["eleccion"] == ronda["modelo"]
    st.write(f"Elegiste **{MODELOS[respuesta['eleccion']]}**. "
             f"El generador fue **{MODELOS[ronda['modelo']]}**. "
             + ("¡Acertaste el modelo!" if acierto else "Veamos qué pistas daban los datos."))
    columnas = st.columns(3)
    columnas[0].metric("Media observada", f"{media:.3f}")
    columnas[1].metric("Varianza observada", f"{varianza:.3f}")
    columnas[2].metric("Varianza / media", f"{indice:.3f}" if media > 0 else "No definido")
    st.write(f"Estimaste **{respuesta['estimacion']:.3f}** mutaciones por región. "
             f"La media real era **{ronda['media']:.3f}**: "
             f"tu error relativo fue **{100 * respuesta['error']:.1f} %**.")
    st.info("La media es un promedio ponderado: multiplicá cada conteo por su "
            "frecuencia y sumá esos productos. La barra más alta es la moda; "
            "no necesariamente es la media. La media observada estima la real "
            "y puede diferir por azar. El puntaje usa la real, como en el notebook.")
    if ronda["modelo"] == 1:
        varianza_teorica = ronda["media"]
        st.write("**Enseñanza:** en Poisson todas las réplicas comparten una tasa. "
                 "Media y varianza teóricas son iguales, aunque las observadas "
                 "pueden diferir por muestreo. Un índice cercano a 1 es compatible con Poisson.")
    else:
        alfa = ronda["alfa"]
        varianza_teorica = ronda["media"] + ronda["media"] ** 2 / alfa
        st.write("**Enseñanza:** en la binomial negativa las tasas varían entre réplicas. "
                 "Esa heterogeneidad agrega dispersión: varianza = μ + μ²/α. "
                 "Puede generar muchos ceros y una cola larga. Para una misma media, "
                 "cuanto menor es α, mayor es la dispersión.")
        st.write(f"En esta ronda, α = **{alfa:.3f}**.")
    st.write(f"Varianza teórica: **{varianza_teorica:.3f}**. "
             f"Tasa real por nucleótido y período: **{ronda['tasa_sitio']:.5f}**. "
             f"Tu estimación por nucleótido: **{respuesta['estimacion'] / len(ronda['secuencia']):.5f}**.")
    mostrar_interpretacion_biologica(ronda)
    mostrar_validacion(ronda)


def mostrar_resumen():
    historial = st.session_state.historial
    total = sum(r["puntos"] for r in historial)
    aciertos = sum(r["puntos_modelo"] == 40 for r in historial)
    error_medio = np.mean([r["error"] for r in historial])
    st.subheader(f"Partida terminada: {total}/{TOTAL_RONDAS * 100} puntos")
    st.table([{"Ronda": i, "Media estimada": round(r["estimacion"], 3),
               "Error (%)": round(100 * r["error"], 1),
               "Modelo acertado": "Sí" if r["puntos_modelo"] else "No",
               "Puntos": r["puntos"]} for i, r in enumerate(historial, 1)])
    st.write(f"Identificaste el generador en **{aciertos} de {TOTAL_RONDAS} rondas**. "
             f"Tu error relativo promedio al estimar la media fue **{100 * error_medio:.1f} %**.")
    if aciertos < TOTAL_RONDAS:
        st.write("Para practicar la elección del modelo, compará la dispersión además "
                 "del centro: Poisson tiene varianza teórica igual a la media; "
                 "la binomial negativa permite una varianza mayor.")
    if error_medio >= 0.15:
        st.write("Para mejorar tus estimaciones, considerá las barras de la cola: "
                 "aunque sean bajas, los conteos altos empujan la media hacia arriba.")
    st.info("Dos conjuntos pueden tener medias parecidas y dispersiones muy diferentes. "
            "El puntaje resume estas tres rondas; no mide formalmente tu aprendizaje.")
    st.markdown("**Lo que te llevás sobre biología de mutaciones**")
    st.write("La media describe cuántas mutaciones se esperan en una región durante "
             "un período. Para comparar regiones hay que considerar su longitud y "
             "el tiempo observado. Poisson representa una tasa compartida; la "
             "binomial negativa permite que esa tasa varíe entre copias. Ninguno "
             "de estos modelos determina si las mutaciones afectan la función "
             "de un gen: para eso se necesitan sus posiciones y consecuencias.")


def main():
    st.set_page_config(page_title="Juego de mutaciones", page_icon="🧬", layout="centered")
    st.title("🧬 El juego de las mutaciones")
    st.write("Observá los datos, estimá su media y descubrí qué modelo los generó.")
    with st.expander("Reglas y pistas"):
        st.write("Son 3 rondas de 500 réplicas. Ganás hasta 60 puntos por estimar "
                 "la media real y 40 por identificar el modelo. Los puntos de "
                 "la media bajan linealmente hasta 0 con un error del 30 %.")
        st.write("Poisson: media y varianza teóricas iguales. Binomial negativa: "
                 "varianza mayor que la media. Observá el centro, los ceros y la cola. "
                 "La secuencia es contexto visual y no determina los conteos.")
    if "ronda" not in st.session_state:
        iniciar_partida()
    if st.button("Nueva partida"):
        iniciar_partida()
        st.rerun()
    historial = st.session_state.historial
    # Al resolver, el historial ya incluye la ronda que se sigue mostrando.
    numero = len(historial) if st.session_state.respuesta else len(historial) + 1
    st.progress(len(historial) / TOTAL_RONDAS)
    st.write(f"**Ronda {numero} de {TOTAL_RONDAS}** · "
             f"Acumulado: {sum(r['puntos'] for r in historial)} puntos")
    ronda = st.session_state.ronda
    st.write(f"500 réplicas independientes · {len(ronda['secuencia'])} nucleótidos "
             "por réplica · mismo período de observación")
    with st.expander("Ver la secuencia de referencia"):
        st.code(ronda["secuencia"], language=None, wrap_lines=True)
    mostrar_histograma(ronda["datos"])
    if st.session_state.respuesta is None:
        with st.form(f"respuesta_{len(historial)}"):
            texto = st.text_input("¿Cuál es la media de mutaciones por región?",
                                  placeholder="Ejemplo: 2,5")
            eleccion = st.radio("¿Qué modelo parece más adecuado?", [1, 2],
                                format_func=MODELOS.get, index=None)
            enviar = st.form_submit_button("Resolver y aprender")
        if enviar:
            try:
                estimacion = float(texto.strip().replace(",", "."))
                respuesta = evaluar_respuesta(ronda, estimacion, eleccion)
            except (ValueError, OverflowError):
                st.error("Ingresá una media válida (finita y mayor o igual a cero) "
                         "y seleccioná un modelo.")
            else:
                st.session_state.respuesta = respuesta
                historial.append(respuesta)
                st.rerun()
    else:
        mostrar_resultado(ronda, st.session_state.respuesta)
        if len(historial) < TOTAL_RONDAS:
            if st.button("Siguiente ronda", type="primary"):
                st.session_state.ronda = generar_ronda(st.session_state.rng)
                st.session_state.respuesta = None
                st.rerun()
        else:
            mostrar_resumen()


if __name__ == "__main__":
    main()
