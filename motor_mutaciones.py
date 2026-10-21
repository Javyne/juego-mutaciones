"""Simulación del notebook, separada de la interfaz para facilitar su lectura."""
import numpy as np
from scipy import stats

TOTAL_RONDAS = 3
MODELOS = {1: "Poisson", 2: "Binomial negativa"}


def generar_ronda(rng):
    """Sortea el modelo, la región y los 500 conteos."""
    modelo = int(rng.choice([1, 2]))
    longitud = int(rng.integers(50, 501))
    secuencia = "".join(rng.choice(list("ATCG"), size=longitud))
    tasa_sitio = float(rng.uniform(0.004, 0.02))
    media = tasa_sitio * longitud
    alfa = None if modelo == 1 else float(rng.uniform(0.5, 2.5))
    if alfa is None:
        datos = rng.poisson(media, size=500)
    else:
        tasas = rng.gamma(shape=alfa, scale=media / alfa, size=500)
        datos = rng.poisson(tasas)
    return dict(datos=datos, modelo=modelo, secuencia=secuencia,
                tasa_sitio=tasa_sitio, media=media, alfa=alfa)


def evaluar_respuesta(ronda, estimacion, eleccion):
    """Hasta 60 puntos por estimar la media y 40 por acertar el modelo."""
    if not np.isfinite(estimacion) or estimacion < 0:
        raise ValueError("La estimación debe ser finita y mayor o igual a cero.")
    if eleccion not in MODELOS:
        raise ValueError("Elegí un modelo.")
    error = abs(estimacion - ronda["media"]) / ronda["media"]
    puntos_media = max(0, round(60 * (1 - error / 0.30)))
    puntos_modelo = 40 if eleccion == ronda["modelo"] else 0
    return dict(estimacion=estimacion, eleccion=eleccion, error=error,
                puntos_media=puntos_media, puntos_modelo=puntos_modelo,
                puntos=puntos_media + puntos_modelo)


def chi2_bondad_ajuste(datos, distribucion, parametros_estimados=1):
    """Agrupa categorías contiguas hasta reunir al menos 5 casos esperados."""
    maximo = int(datos.max())
    observados = np.bincount(datos, minlength=maximo + 1).astype(float)
    esperados = len(datos) * distribucion.pmf(np.arange(maximo + 1))
    esperados[-1] += len(datos) * distribucion.sf(maximo)
    grupos = []
    obs_acum = esp_acum = 0.0
    for observado, esperado in zip(observados, esperados):
        obs_acum += observado
        esp_acum += esperado
        if esp_acum >= 5:
            grupos.append([obs_acum, esp_acum])
            obs_acum = esp_acum = 0.0
    if esp_acum > 0:
        if grupos:
            grupos[-1][0] += obs_acum
            grupos[-1][1] += esp_acum
        else:
            grupos.append([obs_acum, esp_acum])
    estadistico = sum((obs - esp) ** 2 / esp for obs, esp in grupos)
    gl = len(grupos) - 1 - parametros_estimados
    p = float(stats.chi2.sf(estadistico, gl)) if gl > 0 else float("nan")
    return estadistico, gl, p


def modelos_para_validar(ronda):
    """Como en el notebook: media estimada y, en NB, alfa real revelado."""
    media = float(ronda["datos"].mean())
    ajustes = [("Poisson", stats.poisson(mu=media))]
    if ronda["alfa"] is not None:
        alfa = ronda["alfa"]
        ajustes.append(("Binomial negativa", stats.nbinom(
            n=alfa, p=alfa / (alfa + media))))
    return ajustes
