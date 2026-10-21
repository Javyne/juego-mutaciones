# Juego de mutaciones

Interfaz sencilla en Python basada en juego_mutaciones.ipynb.
El notebook original se conserva sin cambios.

## Abrir el juego

Abrí una terminal en esta carpeta e instalá las dependencias una vez:

    python -m pip install -r requirements.txt

Después ejecutá:

    python -m streamlit run app_mutaciones.py

Se abre en el navegador. Si no se abre automáticamente, entrá a
http://localhost:8501. Para detenerlo, presioná Ctrl+C en la terminal.
No hace falta ejecutar el notebook ni tener Jupyter abierto.

## Cómo jugar

1. Observá el histograma inicial de 500 réplicas; podés desplegar la secuencia.
2. Escribí la media de mutaciones por región (admite coma o punto decimal)
   y elegí Poisson o binomial negativa.
3. Presioná Resolver y aprender: aparecen puntaje, enseñanzas, frecuencias
   observadas frente al modelo y QQ-plots con interpretación del test chi-cuadrado.
4. Avanzá a la siguiente ronda. Al terminar tres rondas aparece un resumen
   con recomendaciones según tus respuestas. Nueva partida reinicia el juego.

## Organización del código

- motor_mutaciones.py: generación de datos, puntaje y validación estadística.
- app_mutaciones.py: pantallas, preguntas, gráficos y explicaciones.
- requirements.txt: bibliotecas necesarias.

Se conservan los rangos, modelos y puntaje del notebook. Cada partida usa un
generador aleatorio nuevo. Los datos no cambian al interactuar con los controles.
Al resolver se bloquea la respuesta, para no duplicar puntos.

La media observada estima la real, pero puede diferir por azar. El puntaje usa
la media real. En rondas NB, la validación usa el alfa real revelado y estima
solo la media, como en el notebook. Los p-valores son aproximados y un valor
alto no demuestra que el modelo sea verdadero.

## Interpretación biológica

Cada resolución traduce la media a eventos por copia y a un ejemplo
de 100 copias, explica la tasa por nucleótido y distingue una tasa
común (Poisson) de tasas heterogéneas (binomial negativa). Incluye
el porcentaje observado de copias sin mutaciones, hipótesis sobre
replicación, reparación y exposición, y los límites para inferir
efectos funcionales. Las fuentes están enlazadas dentro del juego.
