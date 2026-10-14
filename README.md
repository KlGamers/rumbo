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

El núcleo solo necesita `numpy` y `httpx`. Agrega `[static]` para
`model2vec` u `[onnx]` para modelos transformer con `fastembed`.

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
| `LinearClassifier` | Regresión logística entrenada en NumPy. La opción más precisa y la más rápida |
| `LlmClassifier` | Le pide a un modelo de chat un veredicto en JSON y lo interpreta con cuidado |
| `HashEmbedder` | Hashing de n-gramas de caracteres con pesos IDF. No descarga ningún modelo |
| `StaticEmbedder` | Embeddings estáticos con `model2vec` |
| `OnnxEmbedder` | Modelos tipo transformer en ONNX con `fastembed`, sin GPU |
| `ConcatEmbedder` | Une varios embedders en un solo vector |
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
consultas fuera de alcance a una ruta de respaldo. Las rutas se construyen con
las 15,000 consultas de entrenamiento más las 100 de ejemplo fuera de alcance.
El umbral y la regularización se eligen con las 3,100 consultas de validación,
y la tabla reporta las 5,500 de prueba, de las cuales 1,000 están fuera de
alcance. La latencia es por mensaje, en una laptop Apple Silicon, e incluye el
cálculo del embedding.

| router | exactitud | en alcance | recall fuera de alcance | F1 macro | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| centroide / hash | 0.6913 | 0.8449 | 0.0000 | 0.6966 | 0.09 | 0.12 |
| kNN / hash | 0.8116 | 0.9313 | 0.2730 | 0.8251 | 9.02 | 11.44 |
| kNN / estático | 0.8180 | 0.9473 | 0.2360 | 0.8313 | 0.47 | 0.82 |
| lineal / hash | 0.8715 | 0.9622 | 0.4630 | 0.8868 | 0.14 | 0.17 |
| lineal / transformer + hash | 0.8925 | 0.9760 | 0.5170 | 0.9066 | 4.67 | 6.92 |

Qué dicen los números:

- El mejor router acierta el **97.6% de los mensajes dentro de alcance**. Usa
  un clasificador lineal entrenado sobre la concatenación de embeddings de
  `bge-small-en-v1.5` (ONNX, corre local) y n-gramas de caracteres.
- Sin descargar ningún modelo, el clasificador lineal sobre n-gramas llega a
  96.2% y decide en 0.14 ms por mensaje, 60 veces más rápido que kNN sobre las
  mismas características.
- Detectar mensajes fuera de alcance sigue siendo lo más difícil. El mejor
  router atrapa el 52%, contra 27% de kNN. Por eso la exactitud total es 89.3%
  y no 97%: el umbral se ajusta con validación, donde solo 100 de 3,100 casos
  están fuera de alcance.
- Los centroides son los más baratos, pero no separan nada fuera de alcance y
  pierden 13 puntos dentro de alcance.

`LinearClassifier` es regresión logística multinomial entrenada con Adam en
NumPy, sin scikit-learn. `LlmClassifier`, `HttpEmbedder` y `OpenAICompat`
están cubiertos con pruebas sobre transportes simulados, pero no se han medido
contra una API real.

Para reproducirlo:

```sh
pip install -e ".[dev,static,onnx]"
python scripts/bench_clinc.py
```

El script descarga CLINC150 (CC BY 3.0) en `data/`, que no se sube al
repositorio, y el modelo ONNX en la caché de fastembed.

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
pytest -q --cov=rumbo
ruff check . && ruff format --check .
```

La cobertura es de 99% y el CI falla si baja de 95%.

## Licencia

MIT. CLINC150 es de Larson et al., 2019, y no se redistribuye aquí.
