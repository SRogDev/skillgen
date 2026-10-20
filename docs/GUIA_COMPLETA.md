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

**Qué se hizo.** `src/skillgen/evaluation/compare.py`: comparación
**pareada** entre dos archivos de predicc
...[truncated 7249 chars]