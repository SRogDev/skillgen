# SkillGen — Guía completa de aprendizaje

*Qué se construyó, por qué, y cómo defenderlo. Escrita para Roger: cada
fase cierra con lo que debes poder explicar en una entrevista y las
preguntas duras que te harán, con la respuesta honesta que este repo
respalda. Todos los números salen de artefactos reales del repo
(`experiments/`, `STATUS.md`, `docs/FINDINGS.md`) — nada está inventado.
Lo marcado **SMOKE** es prueba de tubería en CPU con un modelo de 0.5B,
no un resultado del modelo real. Los resultados reales del 7B **aún no
existen**: corren en Kaggle (ver el runbook al final).*

---

## La idea en un párrafo

SkillGen especializa un LLM pequeño (7B, QLoRA) para que, dada una
petición en lenguaje natural, genere un **JSON Skill Blueprint** que un
harness valida y materializa como una Agent Skill real en disco. El
proyecto no es "hacer fine-tuning": es recorrer el ciclo completo de un
ML Engineer — datos, fundamentos, baselines, entrenamiento, evaluación
con estadística, ablaciones, serving y export — con evidencia en cada
paso y sin afirmar nada que no se haya medido.

---

## Fase 0 — EDA y entorno: conoce tus datos antes de entrenar

**Qué se hizo.** Se tokenizó el texto de entrenamiento *completo y
renderizado* (system prompt de PLAN.md §8 + instrucción + blueprint,
con el chat template nativo de Qwen2.5-7B-Instruct) sobre los splits
congelados 800/100/100, y se midió todo: `experiments/eda_v1/`,
`notebooks/00_environment.ipynb`, `notebooks/01_dataset_eda.ipynb`,
código testeado en `src/skillgen/data/eda.py`.

**Lo que salió (esto es lo que defiendes):**
- La distribución de longitudes es brutalmente asimétrica: train
  **p50 = 3.838 tokens, p95 = 34.071, máximo = 1.947.918** — el
  registro más largo ni siquiera cabe en la ventana de contexto del
  modelo (131.072). El **46,75% del train (374/800) supera 4.096
  tokens**.
- El dataset está *sano*, no roto: **0 fallos de parseo JSON y 0
  blueprints sin `SKILL.md`** en los 1.000 registros. El problema es
  longitud, no validez.
- **Decisión: `max_seq_length = 8192` y filtrar, nunca truncar.**
  E1 entrena con los **582 registros** que caben enteros. ¿Por qué no
  truncar? Porque un blueprint truncado es JSON inválido: entrenarías
  al modelo a producir salidas incompletas. Val/test **nunca** se
  filtran para evaluar (solo se marcan los que exceden: 29/100 en
  cada uno) — si no, inflarías tus métricas escondiendo los casos
  duros.

**El concepto.** En ML, la decisión más cara es la que tomas sin medir.
Un `max_seq_length` copiado de un tutorial (4.096) habría destruido
casi la mitad de tus datos en silencio.

**Preguntas de entrevista:**
- *"¿Por qué 8192 y no 4096 o 16384?"* → Porque el EDA lo dice: con
  4096 pierdes el 46,75% (truncado) y con 16384 pagas memoria y
  cómputo por el ~13% más largo, que además es atípico; 8192 conserva
  582 ejemplos completos y cabe en una T4 de 16 GB con QLoRA.
- *"¿Filtrar no sesga el dataset?"* → Sí, y está documentado como
  coste conocido: las skills multi-archivo muy grandes quedan
  infrarrepresentadas. La alternativa honesta era partir blueprints,
  que rompe la unidad de aprendizaje; se acepta el sesgo y se declara.

---

## Fase 1 — Fundamentos: nada de cajas negras

**Qué se hizo.** `notebooks/02_pytorch_foundations.ipynb`: un LM de
caracteres diminuto (611.015 parámetros) entrenado con un **loop
manual** (`tokenize → forward → loss → backward → step`) sobre 3
blueprints reales: loss **5,60 → 1,04** en 300 pasos (memorización de
un corpus minúsculo — sirve para *entender* el loop, no es un
resultado de capacidad). `notebooks/03_lora_from_scratch.ipynb` +
`src/skillgen/training/fundamentals.py`: **LoRA implementado a mano**
(envolver un `Linear` con matrices A/B, base congelada), con tests.

**Lo que defiendes:**
- LoRA: ΔW = B·A con rango r ≪ d. A con init kaiming, **B en cero →
  ΔW = 0 al inicio** (verificado exactamente): el modelo arranca
  siendo el modelo base. Parámetros entrenables por capa =
  **r·(in + out)**.
- La matemática de memoria que justifica QLoRA: fine-tuning completo
  de un 7B ≈ **84 GB** de estados (pesos + gradientes + Adam) frente
  a **QLoRA ≈ 3,66 GB** (base en 4-bit NF4 + adaptadores fp16:
  40.370.176 parámetros con r=16/all-linear/28 capas + optimizador de
  8-bit) — **~22,9× menos**. Por eso el entrenamiento real cabe en
  una T4 gratuita de 16 GB.
- En el smoke de LoRA (10 pasos), los pesos base quedaron
  **bit-idénticos** (`torch.equal`): LoRA jamás toca el modelo base;
  eso es lo que hace seguro experimentar.

**Preguntas de entrevista:**
- *"¿Por qué LoRA y no fine-tuning completo?"* → Memoria (84 GB vs
  3,66 GB) y reversibilidad: el adaptador es un archivo de ~100 MB
  separable del base. Con 582 ejemplos, además, entrenar 7B completos
  es invitar al overfitting.
- *"¿Qué hace exactamente el masking assistant-only?"* → Pone
  `labels = -100` en system/user para que la loss solo se calcule
  sobre la respuesta. Sin eso, el modelo gasta capacidad aprendiendo
  a predecir las instrucciones del usuario (Fase 3 lo verifica
  decodificando, y E2 lo mide como ablación).

---

## Fase 2 — Baselines: sin baseline, el fine-tune no demuestra nada

**Qué se hizo.** Pipeline E0 completo (`src/skillgen/evaluation/
baseline.py`, CLI `scripts/run_baseline.py`): construye los prompts
(zero-shot y few-shot con **3 ejemplares fijos del split train**,
verificado por test que no aparecen en val/test), genera predicciones
y puntúa **reutilizando el harness existente**
(`scripts/eval_harness.py`, capas L1–L5) — no se reimplementó ninguna
métrica. `notebooks/04_baseline.ipynb`.

**SMOKE (prueba de tubería, NO es E0):** Qwen2.5-0.5B zero-shot sobre
5 ejemplos de val en CPU. Las 5 salidas fueron JSON coherente y
temático, pero con **esquemas inventados**: a todas les faltan
`blueprint_version`, `purpose`, `when_to_use`, `when_not_to_use` y
`files`, y usan nombres Title/CamelCase en vez del kebab-case de
referencia. Puntuación: **L1 1,0000 / L2 0,0000 / L3 0,1290 /
L5 0,0000 → SkillGen Score 0,3245** (renormalizado sin L4). Fallos:
**E2 (campo requerido ausente) ×5 y E4 (nombre inválido) ×5**.

**El concepto.** El hallazgo del smoke es el argumento del proyecto
entero: *el modelo sin entrenar ya "sabe JSON", pero inventa el
esquema* — la invención de esquema es el fallo dominante, y eso es
exactamente lo que el fine-tuning debe corregir. Cuando E1 corra, la
comparación pareada contra E0 dirá si lo corrigió o no.

**Preguntas de entrevista:**
- *"¿Y si el few-shot ya resuelve el formato?"* → Esa es la hipótesis
  de E0b, y por eso existe: si few-shot iguala al fine-tune, el
  fine-tune no aporta. E0b está implementado y testeado; su corrida
  real es parte del runbook de Kaggle.
- *"¿Por qué renormalizar el Score sin L4?"* → L4 es el juez LLM
  (necesita API key); sin juez, los pesos de §33 se renormalizan
  sobre 0,85 y se reporta también la suma sin renormalizar. Ocultar
  que falta una capa sería deshonesto.

---

## Fase 3 — Entrenamiento QLoRA: el código y sus puertas de calidad

**Qué se hizo.** `src/skillgen/training/prepare.py` (renderiza con el
chat template nativo y construye labels assistant-only),
`train.py` (TRL `SFTTrainer`, dirigido por YAML), `config.py`, y
`configs/`: E1 (§18 exacto: r16/α32, NF4 + doble cuantización, lr
2e-4, 2 epochs, batch efectivo 16, warmup 3%, cosine, adamw_8bit,
seed 42) más **una variante por ablación** (E2 sin masking, E3
epochs 1/3, E4 rango 8/32, E5 solo atención, E6 datos 250/500) — un
test garantiza que cada config difiere de E1 *solo* en su variable.
`notebooks/05_qlora_training.ipynb` es el notebook de producción para
Kaggle (instala unsloth en la celda; no es dependencia local porque
fija su propia pila de torch y bitsandbytes necesita CUDA).

**La puerta que importa (PLAN §21):** antes de entrenar, el código
decodifica los labels ≠ -100 del primer ejemplo e imprime el texto:
en el smoke salió **exactamente el blueprint + `<|im_end|>`**, nada
más. Un template mal aplicado enmascara todo en silencio y obtienes
`loss = 0.0` sin aprender nada — esta verificación es la diferencia
entre "corrió" y "entrenó".

**SMOKE (no es E1):** 0.5B con LoRA plano r8 (sin cuantización: no
hay GPU), 20 registros filtrados, 10 pasos en CPU: train loss
**2,4453 → 1,8009**. Prueba la tubería (filtrado, masking, TRL,
guardado del adaptador); no dice nada de la calidad del 7B.

**Hallazgo de ingeniero, no de tutorial:** se verificó *inspeccionando
TRL 1.16* que `SFTTrainer` acepta una columna `labels` propia sin
re-enmascarar — por eso el masking se construye en `prepare.py` y
jamás se delega al comportamiento interno de una librería.

**Preguntas de entrevista:**
- *"¿Por qué QLoRA y no LoRA fp16?"* → En una T4 de 16 GB, el base
  fp16 (~14 GB) no deja sitio para activaciones ni optimizador; el
  base en 4-bit (~3,5 GB) sí. La pérdida por cuantizar el base es
  pequeña porque los adaptadores se entrenan en fp16.
- *"¿Cómo sabes que tu masking funciona?"* → Decodifico los labels
  del primer ejemplo antes de entrenar y lo muestro: solo el turno
  del asistente. Está en el código como puerta obligatoria, no como
  comprobación manual ocasional.

---

## Fase 4 — Evaluación seria: estadística, fallos y olvido

**Qué se hizo.** `src/skillgen/evaluation/compare.py`: comparación **pareada** entre dos archivos de predicciones — no compares promedios a ciegas, compara *qué cambió por ítem*: cuántos se arreglaron, cuántos se rompieron. Conteo pareado (both/only-A/only-B/both-fail sobre l1_parse, l2_valid o l5), McNemar exacto (binomial de dos colas vía `math.comb` — stdlib, sin scipy), intervalos de Wilson para las tasas, y bootstrap pareado (seed 42, 2000 resamples) para el intervalo de la diferencia de SkillGen Score. Decisión documentada: el bootstrap corre sobre la diferencia de métricas binarias porque las contribuciones por ítem del Score no se guardan en `scores.json`. `render_comparison_md` emite el esqueleto §47 BEST/WORST/REGRESSION/NEW-FAILURE. 15 tests (RED → GREEN), incluido el McNemar calculado a mano: b=9, c=1 → p = 0.021484375.

**Por qué pareada y no dos promedios.** Con n=100 en el test, una mejora de +2 puntos puede ser ruido. McNemar te dice si los cambios por ítem son sistemáticos; el bootstrap te da el intervalo de la diferencia. Es la diferencia entre "subió 2 puntos" y "subió 2 puntos y el intervalo del 95% no cruza cero".

**Análisis de fallos.** `scripts/failure_analysis.py` mapea las señales del harness a la taxonomía §48 (E1–E10), con el mapeo y las lagunas honestas en `docs/ERROR_TAXONOMY.md`: el repo solo nombra los anclajes (E1 estructura inválida, E10 fallo de formato); E2–E9 son una operacionalización local documentada; un fallo de L5 aguas abajo de un L2 no se cuenta dos veces como E8. El primer archivo real de fallos (`experiments/e0_smoke/failures.md`): **E2 (campo requerido ausente) ×5, E4 (nombre inválido) ×5** — todos los ítems del smoke inventan el esquema (5 campos ausentes + nombres no-kebab). Ese es exactamente el fallo que el SFT debe corregir: la hipótesis de la Fase 2, escrita como taxonomía medible.

**Chequeo de olvido (forgetting).** `src/skillgen/evaluation/forgetting.py`: antes/después por loglikelihood (argmax sobre 4 opciones, sin generación — barato y determinista). `load_arc_questions()` carga ARC-Challenge desde HF cuando hay red y cae al set local original de 20 preguntas (ciencia/historia/mates/geografía/idioma), etiquetado por fuente — el set local **NO es ARC**. SMOKE: base 0.5B **0.8000 (16/20)** vs adaptador del smoke **0.8000 (16/20)**, predicciones idénticas — esperado para un adaptador de 10 pasos; la regla §37 (>5% de caída = hallazgo) aplica a la corrida real (50 preguntas ARC, adaptador E1, Kaggle).

**El registro de experimentos.** `experiments/README.md`: E0a–E7 con estados honestos (E0a solo-smoke, E0b implementado sin ninguna corrida, E1 solo-smoke, E2–E6 listos-pendientes-GPU, E7 solo-diseño) y la puerta §39: un candidato se promueve **solo si** su SkillGen Score ≥ score actual + 2 puntos **y** no hay regresión en el expert set. Si no, se rechaza.

**Preguntas de entrevista:**
- *"¿Por qué McNemar y no un t-test?"* → Porque comparo el mismo ítem antes y después: los datos son pareados y binarios (arreglado/roto). El t-test asume normalidad de diferencias continuas; McNemar es el test exacto para la tabla 2×2 de discordantes.
- *"¿Cómo sabes que no estás overfiteando a tu métrica?"* → Tres defensas: el test está congelado y nunca se toca; el expert set es curaduría humana independiente con rúbrica manual; y el chequeo de olvido mide que no destruí conocimiento general.

---

## Fase 5 — Serving, export y honestidad documentada

**Qué se hizo.** `src/skillgen/serving/app.py`: una API FastAPI (`uv run uvicorn skillgen.serving.app:app`) que implementa el pipeline §42: generate → parse L1 → validación de esquema L2 (**importada de `scripts/eval_harness.py`, nunca reimplementada** — el mismo código que evalúa es el que valida en producción) → reparación alimentando los errores de vuelta al modelo (máx. 2 reintentos) → éxito `200 {blueprint, validation_report}` o rechazo tipado `422 blueprint_invalid` (3 intentos, últimos errores) o `502 backend_error`. Los backends están detrás de un protocolo `ModelBackend`: un `FakeBackend` determinista para tests/demo y un `TransformersBackend` **perezoso** (carga el modelo en el primer `generate()`, nunca al importar — si cargara al importar, ni los tests podrían correr sin GPU), con el system prompt §8 verbatim y los defaults §41 (temperature 0.3, top_p 0.9, max_new_tokens 4096, repetition_penalty 1.1), decodificando solo tokens nuevos, y adaptador opcional vía `SKILLGEN_ADAPTER_PATH`. El backend por defecto es el fake de demo — devuelve un blueprint real corto de train (`larksuite_cli__good-skill`, pasa L2) — **a menos que `SKILLGEN_MODEL_ID` esté seteado**, y `GET /health` siempre nombra el backend activo (`fake-demo (NOT a trained model)` por defecto). Nadie puede confundir la demo con el modelo entrenado: el endpoint lo declara.

**Por qué el loop de reparación.** La Fase 2 demostró que el modelo base genera JSON coherente pero inventa el esquema. En producción no puedes devolver eso: el loop generate → validar → reparar convierte un fallo silencioso en un rechazo tipado. Es la diferencia entre "a veces sale mal" y "sale bien o te dice por qué no".

**Export.** `scripts/export_model.py` implementa el §44 como plan ordenado explícito: cargar base+adaptador → **merge en 16-bit (`merged_16bit`)** → **GGUF `q4_k_m` desde el modelo mergeado** → instrucciones de `ollama create`. Regla dura testeada: **nunca merge directo a 4-bit** (cuantizar el adaptador antes de fusionarlo degrada los pesos que acabas de entrenar). Sin GPU + unsloth, `--dry-run` imprime el plan y los comandos exactos y sale 0; sin `--dry-run` sale 2 con mensaje claro. En esta fase no se exportó nada: no hay adaptador real todavía.

**Honestidad documentada.** `MODEL_CARD.md` (raíz): base PROVISIONAL, **no entrenado a 7B, sin pesos publicados, los resultados reales del 7B no existen** — la única evidencia son los números SMOKE/0.5B etiquetados. `docs/FINDINGS.md` (hallazgos acumulados Fases 0–5) y `docs/DECISIONS.md` (9 ADRs). 8 tests de serving (RED → GREEN: el RED fue `ModuleNotFoundError`), 6 de export. Suite: **86/86 pytest en verde**.

**Preguntas de entrevista:**
- *"¿Por qué no cargas el modelo al arrancar la API?"* → Porque entonces cada test, cada import y cada despliegue necesitaría la GPU y los pesos. Carga perezosa: el proceso arranca barato y el primer request paga el costo — y `/health` te dice qué backend está vivo.
- *"¿Por qué merge en 16-bit antes de cuantizar a GGUF?"* → El adaptador vive en fp16; fusionarlo al base en 16-bit preserva lo aprendido y luego cuantizas el modelo completo una sola vez. Merge directo a 4-bit = cuantizar dos veces = degradación compuesta.

---

## El runbook de Kaggle: las puertas en orden

Cuando pases la key, la secuencia es esta — cada puerta bloquea la siguiente:

1. **Entorno** (`notebooks/05_qlora_training.ipynb`): instala unsloth + bitsandbytes en la celda, asserts de GPU (T4) y CUDA disponible. Sin GPU no hay corrida: la puerta es el assert, no tu palabra.
2. **Modelo base**: sigue **PROVISIONAL**. Antes de congelarlo hay que verificar los candidatos 2026 — Mistral Large 4 publica pesos el 27 de octubre con licencia custom (aviso programado el 2 de noviembre). Si la licencia lo permite, se compara antes de congelar; si no, Qwen2.5-7B queda congelado por decisión documentada en `docs/DECISIONS.md`.
3. **Masking**: el código decodifica los labels ≠ -100 del primer ejemplo e imprime el texto — debe ser **exactamente el blueprint + `<|im_end|>`**. Si sale otra cosa, no entrenas: paras y depuras el template.
4. **E0 antes que E1**: corre el baseline del 7B (zero-shot y few-shot) sobre val/test completos con el harness. Sin baseline, el fine-tune es indefendible: es el número contra el que se mide todo.
5. **E1**: QLoRA con `configs/qlora_e1.yaml` (~73 pasos sobre los 582 registros). Loss de entrenamiento sana orientativa: **0,5–1,0**. Guarda el adaptador.
6. **Comparación pareada** (`src/skillgen/evaluation/compare.py`): E0 vs E1 sobre el test. Lee primero los conteos pareados y McNemar, después el Score. **Puerta §39**: se promueve solo si SkillGen Score ≥ actual + 2 puntos **y** sin regresión en el expert set.
7. **Ablaciones** E2–E6 (una variable por experimento, garantizado por test) — solo si E1 pasa la puerta; si no, el análisis de fallos manda.
8. **Olvido**: 50 preguntas ARC reales con el adaptador E1. Regla §37: caída >5% = hallazgo que se reporta, no se esconde.
9. **Export** (`scripts/export_model.py` sin `--dry-run`): merge 16-bit → GGUF q4_k_m → Ollama. Actualiza `MODEL_CARD.md` con los números reales.

Reglas que no se negocian: el test congelado no se toca ni se entrena con él; W&B solo si hay key; todo resultado negativo se documenta en `docs/FINDINGS.md`.

---

## Lo que solo tú puedes hacer

Dos cosas, y las dos están preparadas para que sean rápidas:

1. **Pasar la key de Kaggle** — cuando la tengas, el notebook 05 corre en un clic. Sin la key, todo lo anterior es tubería probada esperando GPU.
2. **Curar el expert set (PLAN §31)** — 20 casos difíciles: skills de skills.sh especialmente complejas y/o instrucciones trampa redactadas por ti, con rúbrica manual 0–100. Es el conjunto con más peso cualitativo y el veto de la puerta §39. Tienes el andamiaje listo en `data/expert/`: `README.md` (cómo curar cada caso en ~10 minutos, ideas de trampas ligadas a la taxonomía de fallos E1–E10), `_template.E000.json` (el formato exacto), `RUBRICA.md` (5 dimensiones × 20 puntos), y `scripts/validate_expert_set.py` para comprobar el formato. **No redactes los casos por redactar**: cada uno debe ser un caso que tú puedas defender en una entrevista — por eso los curas tú, no un script.

---

*Fin de la guía. El proyecto está completo como código; los números reales del 7B — E0, E1, ablaciones, olvido con ARC-50, export ejecutado — esperan la corrida en Kaggle. Cuando existan, `docs/FINDINGS.md` es donde se anotan.*
