# rumbo

[![ci](https://github.com/KlGamers/rumbo/actions/workflows/ci.yml/badge.svg)](https://github.com/KlGamers/rumbo/actions/workflows/ci.yml)

Router de intenciones para aplicaciones con LLMs, con el arnés de evaluación en
el mismo paquete. Decide qué modelo o herramienta debe atender cada mensaje,
manda a un respaldo cuando nada encaja y te dice qué tan seguido acierta.

## Por qué

Casi toda app que habla con más de un modelo termina con un montón de `if
"code" in msg`, o le pide a un modelo grande que elija la ruta y paga por eso
en cada petición. Las dos opciones son difíciles de medir. rumbo trata el
enrutamiento como un clasificador que se puede calificar: defines rutas con
unos cuantos ejemplos, eliges una estrategia, la corres contra datos
etiquetados y lees la exactitud antes de salir a producción.

## Instalación

```sh
pip install git+https://github.com/KlGamers/rumbo
```

El núcleo solo necesita `numpy` y `httpx`. Agrega `[static]` para el embedder
opcional de `model2vec`.

## Inicio rápido

```python
from rumbo import HashEmbedder, KnnClassifier, Route, Router, Stage

routes = [
    Route("code", examples=("fix this traceback", "write a python script"), target="coder"),
    Route("math", examples=("solve 3x + 7 = 22", "derivative of sin x"), target="reasoner"),
]
emb = HashEmbedder().fit([e for r in routes for e in r.examples])
rtr = Router(routes, [Stage(KnnClassifier(routes, emb, k=2), min_score=0.1)], fallback="general")

dec = rtr.route("why does my script crash")
print(dec.route, round(dec.score, 2), dec.stage, dec.fallback)
```

O guarda las rutas en un archivo y usa la línea de comandos:

```sh
rumbo route "solve 4x = 12" -c examples/routes.toml
rumbo eval examples/sample.jsonl -c examples/routes.toml --min-acc 0.9
```

`rumbo eval` sale con código 1 por debajo de `--min-acc`, así que puede frenar
un pipeline de CI.

## Piezas

| Pieza | Qué hace |
|---|---|
| `PatternClassifier` | Reglas regex por ruta. Gratis, exacto y bueno como primera etapa |
| `CentroidClassifier` | Un vector promedio por ruta. Una sola multiplicación de matrices pequeña |
| `KnnClassifier` | Similitud promedio de los `k` ejemplos más cercanos de cada ruta |
| `LlmClassifier` | Le pide a un modelo de chat un veredicto en JSON y lo interpreta con cuidado |
| `HashEmbedder` | Hashing de n-gramas de caracteres con pesos IDF. No descarga ningún modelo |
| `StaticEmbedder` | Embeddings estáticos con `model2vec` |
| `HttpEmbedder` | Cualquier endpoint `/embeddings` compatible con OpenAI |
| `Router` | Corre las etapas en orden y acepta la primera respuesta con suficiente confianza |
| `Router.dispatch` | Manda el mensaje al proveedor de la ruta y, si falla, a sus respaldos |
| `evals` | Exactitud, F1 macro, precisión y recall por ruta, matriz de confusión y latencia |

### Etapas

Un router es una lista de etapas. Cada una tiene un clasificador, un puntaje
mínimo y un margen mínimo entre la mejor ruta y la segunda. Gana la primera
etapa que cumpla ambos. Si la última no los cumple, el mensaje se va a la ruta
de respaldo.

```python
Router(routes, [
    Stage(PatternClassifier(routes)),
    Stage(KnnClassifier(routes, emb), min_score=0.2, min_margin=0.05),
    Stage(LlmClassifier(routes, ask)),
], fallback="general")
```

### Despacho

```python
from rumbo import OpenAICompat

providers = {
    "coder": OpenAICompat("https://api.example.com/v1", "model-a", key=KEY),
    "general": OpenAICompat("https://api.example.com/v1", "model-b", key=KEY),
}
rep = await rtr.dispatch("why does my script crash", providers)
print(rep.text, rep.provider, rep.tries)
```

Cada ruta define un `target` y `backups` opcionales. Si un proveedor falla o
tarda demasiado, se intenta con el siguiente. `OpenAICompat` reintenta las
respuestas 429 y 5xx con espera exponencial.

## Benchmark

`scripts/bench_clinc.py` enruta las consultas de
[CLINC150](https://github.com/clinc/oos-eval) a sus 10 dominios y manda las
consultas fuera de alcance al respaldo. Las rutas se construyen con las 15,000
consultas de entrenamiento, los umbrales se ajustan con las 3,100 de
validación y la tabla reporta las 5,500 de prueba, de las cuales 1,000 están
fuera de alcance. La latencia es por mensaje, en un núcleo de una laptop Apple
Silicon, e incluye el cálculo del embedding.

| router | exactitud | en alcance | recall fuera de alcance | F1 macro | p50 ms | p95 ms | escalados |
|---|---|---|---|---|---|---|---|
| centroide / hash | 0.6582 | 0.8044 | 0.0000 | 0.6654 | 0.07 | 0.11 | - |
| kNN / hash | 0.8253 | 0.9364 | 0.3250 | 0.8397 | 2.62 | 2.87 | - |
| kNN / hash + ejemplos fuera de alcance | 0.8255 | 0.9364 | 0.3260 | 0.8399 | 2.65 | 2.93 | - |
| cascada / hash: centroide y luego kNN | 0.8225 | 0.9302 | 0.3380 | 0.8369 | 2.63 | 2.85 | 62.0% |
| centroide / estático | 0.6458 | 0.7860 | 0.0150 | 0.6498 | 0.07 | 0.09 | - |
| kNN / estático | 0.8175 | 0.9473 | 0.2330 | 0.8307 | 0.46 | 0.51 | - |
| kNN / estático + ejemplos fuera de alcance | 0.8180 | 0.9473 | 0.2360 | 0.8313 | 0.48 | 0.56 | - |
| cascada / estático: centroide y luego kNN | 0.8155 | 0.9436 | 0.2390 | 0.8287 | 0.54 | 0.60 | 72.8% |

Qué dicen los números:

- kNN sobre n-gramas de caracteres enruta bien el 93.6% de los mensajes en
  alcance sin descargar ningún modelo. Los embeddings estáticos llegan a 94.7%
  y son unas cinco veces más rápidos por mensaje porque sus vectores son más
  pequeños.
- Los centroides son 35 veces más rápidos que kNN y pierden unos 13 puntos en
  alcance.
- Rechazar mensajes fuera de alcance con un umbral de similitud es el punto
  débil. La mejor configuración detecta un tercio. Agregar las 100 consultas
  fuera de alcance de entrenamiento como ruta propia casi no cambia nada.
- Poner centroides antes de kNN no conviene aquí: la mayoría de los mensajes
  escala de todos modos y la exactitud baja un poco. Las etapas están pensadas
  para una última etapa cara, como `LlmClassifier`.

`LlmClassifier`, `HttpEmbedder` y `OpenAICompat` están cubiertos con pruebas
sobre transportes simulados. No se han medido contra una API real.

Para reproducirlo:

```sh
pip install -e ".[dev,static]"
python scripts/bench_clinc.py
```

El script descarga CLINC150 (CC BY 3.0) en `data/`, que no se sube al
repositorio.

## Evaluar tu propio router

Los datos etiquetados van en JSONL con `text` y `route`:

```json
{"text": "what is 15 percent of 80", "route": "math"}
```

```python
from rumbo import evals

rep = evals.run(rtr, evals.load("cases.jsonl"))
print(rep.to_md())
```

El reporte incluye exactitud, F1 macro, exactitud en alcance y recall fuera de
alcance cuando hay respaldo, percentiles de latencia, qué porcentaje decidió
cada etapa y las rutas con peor F1. `best_cut` elige el umbral de puntaje que
maximiza la exactitud en un conjunto de validación.

## Desarrollo

```sh
pip install -e ".[dev]"
pytest -q
ruff check . && ruff format --check .
```

## Licencia

MIT. CLINC150 es de Larson et al., 2019, y no se redistribuye aquí.
