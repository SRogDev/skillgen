# SkillGen — Plan de implementación

**Estado:** Borrador — resolución completa del cuestionario de implementación (58 puntos).
**Pendiente:** 5 decisiones de Roger (ver §0.1). Con esas respuestas se congela v1, se crea el repo y se empieza.

## 0. Decisiones de partida (fijadas por Roger)

- **SkillGen = QLoRA como único entrenamiento ejecutable.** LoRA en fp16 existe solo como notebook
  educativo (`notebooks/04_lora_experiment.ipynb`); no forma parte del pipeline principal (limitación de GPU).
- **Regla de oro del proyecto:** si una decisión no contribuye a aprender ML o a demostrar que
  SkillGen funciona, no pertenece a v1.

### 0.1 Decisiones abiertas (las pregunta Muse)

1. **Hardware:** ¿Kaggle o Colab? → Recomendación: **Kaggle, acelerador T4** (no P100: los kernels
   4-bit actuales de Unsloth/bitsandbytes ya no soportan Pascal sm_60; verificado en reportes de sep-2026).
2. **Idioma del dataset:** ¿inglés, español o mixto? → Recomendación: **inglés** (el formato
   Agent Skills y casi todas las skills reales de referencia están en inglés; el JSON del blueprint
   queda en inglés de todos modos).
3. **Modelo base:** ¿Qwen2.5-7B-Instruct? → Alternativa: Llama-3.1-8B-Instruct.
4. **Curación:** ¿pipeline Claude → validación automática 100% → spot-check humano del 10–15% por lote?
5. **Repo:** ¿público o privado? → Recomendación: **público** (proyecto de aprendizaje, valor de portafolio).

Las secciones marcadas con 🟡 dependen de estas respuestas. Todo lo demás está decidido abajo.

> Nota: el cuestionario proponía rangos de bloques inconsistentes con la numeración de preguntas.
> Aquí se reagrupan de forma coherente: A(1–10), B(11–14), C(15–20), D(21–28), E(29–38), F(39–49), G(50–58).

---

## BLOQUE A — Producto / ML task (1–10)

### 1. Definición de Skill
**Decisión:** Skill = paquete procedimental reutilizable que un agente **lee** para saber cómo hacer
algo bien. Formato: **Agent Skills** (SKILL.md con frontmatter `name`/`description` + instrucciones
imperativas + recursos opcionales). Toda skill generada debe declarar: propósito (1 línea),
`when_to_use` / `when_not_to_use` (listas), instrucciones paso a paso, criterios de calidad y
activación, tools del agente que utiliza (declaradas, no implementadas) y recursos:
`scripts/` (código auxiliar validable), `references/` (conocimiento de apoyo), `templates/`
(esqueletos), `rules/` (restricciones), `assets/` (datos estáticos). El modelo debe aprender el *progressive disclosure* del estándar:
SKILL.md autocontenido y magro (<5000 tokens), detalle en `references/` — una skill que mete todo
en el SKILL.md es una skill mal diseñada.

### 2. Qué NO es una Skill
**Decisión:** tabla de diferenciación en `docs/`. Tool = función ejecutable con schema; Automation =
secuencia programada sin juicio; Workflow = orquestación de pasos; Agent = entidad que decide;
Prompt = texto de una sola invocación; Knowledge base = datos para retrieval; Memory = estado
persistente; Template = esqueleto sin lógica. La skill es la única que empaqueta **criterio
procedimental reutilizable**.

### 3. Unidad exacta que aprende
**Decisión:** v1 = **generación completa desde cero** (SKILL.md + estructura de archivos decidida por
el modelo). La modificación/análisis de skills existentes entra como *input types* del dataset (el
modelo recibe una skill existente en el contexto y genera el blueprint mejorado), no como tareas
separadas.

### 4. Scope de v1
**Decisión:** Entrada: petición corta, instrucción larga, requisitos técnicos, proceso/contexto
empresarial, skill existente. Salida: **Structured Skill Blueprint (JSON) → Harness → filesystem**.
El modelo genera el plano **con los contenidos de los archivos**; el harness valida y materializa.
El modelo nunca toca el filesystem.

### 5. Skill Blueprint (schema exacto)
**Decisión:** JSON con `blueprint_version: "1.0"`.
- **Obligatorios:** `name` (reglas del spec: 1–64 chars, `[a-z0-9-]`, sin guion inicial/final ni
  consecutivos, igual al nombre del directorio), `description` (1–1024 chars según spec; concisa, en
  tercera persona, con WHAT + WHEN + frases de activación — reutilizable como frontmatter),
  `purpose`, `when_to_use[]`, `files[]` (cada uno: `path`, `purpose`, `content`).
- **Opcionales:** `dependencies[]`, `tools[]`, `evaluation_criteria[]`, `metadata{}`,
  `when_not_to_use[]` (las skills reales rara vez lo traen — 70/80 en el piloto; se conserva
  fiel, no se sintetiza).
- **Representación:** scripts/references/templates/rules/assets → entradas en `files[]` con path
  prefijado (`scripts/`, `references/`, …).
- **Validación (harness):** JSON Schema + reglas del estándar Agent Skills: `name`/`description`
  según spec (ver arriba), SKILL.md con frontmatter YAML + cuerpo Markdown (<500 líneas recomendado).
  Reglas de paths: relativos, sin `..`, sin absolutos, máximo 12 archivos; directorios canónicos
  `scripts/`, `references/`, `assets/` (más extras permitidos como `templates/`, `rules/` — el
  estándar los admite). Código permitido solo en `scripts/`; en v1 **no se ejecuta** (solo `ast.parse`).

### 6. Taxonomía del dataset
**Decisión:** se adoptan las 4 dimensiones propuestas — Domain (10), Complexity (simple/medium/complex),
Structure (single_file, multi_file, multi_directory, scripts, references, templates, mixed),
Input type (short_instruction, long_instruction, business_process, technical_requirement,
existing_skill_modification, ambiguous_request, multiple_requirements) — más `language` (en/es)
por si el dataset sale mixto. Suficiente para v1; no se agregan dimensiones hasta que una decisión
de arquitectura las necesite.

### 7. Template exacto de cada ejemplo
**Decisión:** **JSONL**, un objeto por línea.
- Obligatorios: `id`, `instruction`, `expected_blueprint` (**string** con el JSON del blueprint),
  `difficulty`, `domain`, `structure`.
- Opcionales: `context` (incluye `existing_skill` como string cuando aplica), `input_type`,
  `metadata.provenance{source, model, human_reviewed, reviewer}`.
- Multi-turn: fuera de v1 (single-turn). Los metadatos **no** entran al prompt; solo
  `instruction` (+`context`) → prompt, `expected_blueprint` → target. Metadatos reservados para
  análisis y evaluación.

### 8. Chat Template
**Decisión:** chat template **nativo del modelo base** (nada custom; el código lo deriva del
tokenizer, no hardcodeado). System prompt fijo en inglés: *"You are SkillGen, an expert designer of
reusable Agent Skills. Given the user request, output ONLY a valid JSON Skill Blueprint. No
explanations, no markdown fences."* **Loss solo en tokens del assistant** vía
`DataCollatorForCompletionOnlyLM` (response_template derivado del tokenizer); system/user
enmascarados (`labels=-100`). Sin esto el modelo desperdicia capacidad aprendiendo a predecir
instrucciones de usuario.

### 9. ¿Training sobre texto o estructura?
**Decisión:** **JSON Blueprint con file contents** (opción C del cuestionario). JSON puro sin
contenidos no demuestra skills reales; Markdown libre es inparseable. El assistant genera JSON
crudo, **sin code fences**.

### 10. Dataset de errores
**Decisión:** train = **solo outputs correctos** (SFT). Los errores viven en evaluación: el test set
incluye casos adversariales etiquetados con la taxonomía E1–E10 (§48). Se guardan pares good/bad en
formato DPO-ready desde el día 1, pero **sin entrenar DPO en v1**.

---

## BLOQUE B — Dataset (11–14)

### 11. Datos sintéticos vs humanos
**Decisión (Roger):** el 100% de los ejemplos usa **skills reales de skills.sh** como target
(`expected_blueprint`), con **instrucción sintética** (backtranslation: se genera la petición de
usuario que habría originado esa skill). No hay targets sintéticos.
- **Selección (skills.sh tiene 91k+ skills de calidad muy desigual):** filtro por ranking/instals,
  debe pasar `skills-ref validate`, licencia permisiva, y selección **estratificada por dominio**
  (taxonomía §6) para no sesgar a dev-tools. Spot-check humano del 15% de seleccionadas.
- **Instrucción sintética:** generada por Claude, variada en estilo (corta/larga, técnica/negocio,
  ambigua, multi-requisito) para cubrir los input types (§6). **Chequeo de completitud:** la
  instrucción debe contener información suficiente para reconstruir la skill; si no, se regenera o
  se descarta el ejemplo (evita ejemplos inaprendibles y copia literal).
- **Conversión:** `scripts/build_dataset.py` convierte skill-dir → blueprint JSON (frontmatter →
  name/description, cuerpo → purpose/when_to_use/when_not_to_use, archivos → files[] con contenidos).
- **Rol humano:** selección, spot-check de instrucciones, y el expert set (§31).
Cada ejemplo responde "¿de dónde salió?" vía
`metadata.provenance{source: "skills.sh", skill_url, license, instruction_model, human_reviewed}`.

### 12. Data quality
**Decisión:** doble capa. **Automática (100%)**: JSON válido, schema, paths seguros, tests del parser.
**Manual** (rúbrica de ~5 min/ejemplo): 100% del expert set + 15% aleatorio por lote; si un lote baja
de 90% de aprobación, se regenera el lote. Criterios: correctness, completeness, clarity,
consistency, realism, architectural quality, minimality, executability.

### 13. Dataset leakage
**Decisión:** (a) test + expert se generan y **congelan primero**; (b) dedup por firma
(domain, nombre normalizado, structure); (c) similitud de instrucciones con embeddings, umbral
coseno 0.85 → mismo split; (d) prohibido reutilizar templates canónicos entre splits (solo variantes
distintas); (e) una skill de skills.sh aparece en **un solo split** (dedup directo por ID).

### 14. Dataset versioning
**Decisión:** **Git + sha256** por ejemplo y hash global del dataset en v1 (`data/dataset_v1/`,
`CHANGELOG.md`). DVC solo si el peso o los binarios lo exigen.

---

## BLOQUE C — Modelo + QLoRA (15–20)

### 15. Base Model 🟡
**Decisión propuesta:** **Qwen2.5-7B-Instruct** — Apache 2.0, 32k de contexto, chat template nativo,
Unsloth ✓, 4-bit ✓, multilingüe fuerte, excelente en output estructurado. **Instruct sí**: ya obedece
formato; el SFT lo especializa en skills. Alternativa: Llama-3.1-8B-Instruct.

### 16. Hardware 🟡
**Decisión propuesta:** **Kaggle, acelerador T4** (gratis: sesiones ~9–12h, 30h/semana, datasets
versionados). **No P100**: los kernels 4-bit actuales de Unsloth/bitsandbytes ya no compilan para
Pascal (sm_60) — el run muere con `CUDA error: no kernel image is available`. El notebook
`00_environment.ipynb` verifica GPU/VRAM/CUDA/torch/unsloth al inicio y hace assert de
compute capability ≥ 7. Fallback si la GPU asignada es inferior: `max_seq_length` 2048, batch 1,
más accumulation. Sin GPU: no se entrena (solo demo de inferencia en CPU).

### 17. QLoRA
**Decisión:** NF4 + double quantization; compute dtype **fp16** en T4 (bf16 solo si Ampere+).
Target modules: **todos los lineales** (`q,k,v,o,gate,up,down`). **r=16, alpha=32, dropout=0**
(0.05 solo si aparece overfitting), bias="none".

### 18. Hiperparámetros (baseline E1)
```yaml
model: qwen2.5-7b-instruct (unsloth 4-bit)
lora: {r: 16, alpha: 32, dropout: 0.0, bias: none}
quant: {bits: 4, type: nf4, double_quant: true, compute_dtype: fp16}
training:
  lr: 2e-4
  epochs: 3
  per_device_batch: 2
  grad_accum: 4            # effective batch = 8
  warmup_ratio: 0.03
  weight_decay: 0.01
  optimizer: adamw_8bit    # paged_adamw_8bit si hay OOM
  scheduler: cosine
  max_seq_length: 4096    # confirmar con p95 del dataset (notebook 03)
  gradient_checkpointing: true
  max_grad_norm: 1.0
  seed: 42
```
~300 steps totales (800 ejemplos / batch 8 × 3 epochs). Suficiente para QLoRA en tarea estructurada.

### 19. Experimentos (un cambio por vez)
| ID | Cambio vs anterior | Hipótesis |
|----|--------------------|-----------|
| E0 | Baseline: base + few-shot, sin entrenar | punto de comparación honesto |
| E1 | QLoRA config inicial (§18) | mejora > baseline |
| E2 | r=32 | más capacidad → mejor en arquitecturas complejas |
| E3 | lr=1e-4 | estabilidad vs velocidad |
| E4 | dataset v2 (más datos o filtrado) | calidad de datos > hiperparámetros |
Mismo seed, mismo test, mismo juez → causalidad atribuible.

### 20. LoRA educativo
`notebooks/04_lora_experiment.ipynb`: inspecciona adapters, cuenta parámetros entrenables, forward
manual, cálculo de VRAM fp16 vs 4-bit, 10 steps de smoke test. Conclusión documentada: por qué el
entrenamiento completo va en QLoRA y no en LoRA fp16.

---

## BLOQUE D — Training (21–28)

### 21. Training loop
**Decisión:** **TRL SFTTrainer** para el entrenamiento real + notebook con **training loop manual
reducido** (~50 líneas: tokenize → forward → loss → backward → step) para aprender PyTorch de verdad.
Responsabilidades: `src/skillgen/data/` construye el dataset; tokenización con el chat template en
`prepare_dataset`; collator = `DataCollatorForCompletionOnlyLM`; loss en SFTTrainer; evaluación
periódica; checkpoints en `experiments/<id>/checkpoints/`.

### 22. Loss
Cross-entropy **solo sobre tokens del assistant** (§8). Train loss ↓ = mejor predicción de
blueprints. Val loss ↑ con train ↓ = overfitting/memorización (vigilar desde epoch 2 con solo 800
ejemplos). Documentado en el notebook 05.

### 23. Sequence length
**Decisión:** medir con el tokenizer real (media, p50/p95/p99, max) en el notebook 03;
`max_seq_length` = p95 redondeado (estimado 4096). `truncation=True` con log de truncados;
**`packing=False`** en v1 (ejemplos largos y heterogéneos; el packing complica el enmascarado del
assistant).

### 24. Batch
**Decisión:** per_device=2 × accum=4 × 1 GPU → **effective batch = 8**.

### 25. Learning rate
**Decisión:** 2e-4 inicial, cosine, warmup 3%. Cambiar **solo con evidencia**: divergencia temprana
→ 1e-4; estancamiento >100 steps → 3e-4 (raro en QLoRA).

### 26. Epochs
**Decisión:** 3 inicial; **early stopping** (paciencia = 1 evaluación) sobre val loss, evaluando cada
1/3 de epoch.

### 27. Checkpointing
**Decisión:** cada 1/3 de epoch, máx 3 checkpoints, best por val loss. Se versionan **adapters**
(`adapter_model.safetensors` + config), no el modelo completo. `resume_from_checkpoint` soportado.

### 28. Experiment Tracking
**Decisión:** W&B por run: model, dataset_version+hash, git commit, hiperparámetros, GPU/VRAM,
tokens/s, train/val loss, checkpoint path, eval scores. ID: `skillgen-qlora-v1-e{1..n}`.

---

## BLOQUE E — Evaluación (29–38)

### 29. Evaluación automática
**Decisión:** determinista para lo estructural (JSON válido, schema, paths, campos requeridos).
Heurísticas + juez para calidad/arquitectura (nº archivos vs complejidad pedida, presencia de
`when_not_to_use`, longitud de SKILL.md, penalización de complejidad innecesaria).

### 30. LLM-as-Judge
**Decisión:** Claude como juez, temp 0, rúbrica 1–10 por dimensión con anclas, **blind** (no sabe qué
modelo generó qué), orden aleatorio. 1 pasada en test (100) + 3 pasadas en expert (promedio).
Anti-sesgo: el juez nunca evalúa outputs de su propia familia como "mejores por defecto"; la rúbrica
manda.

### 31. Evaluación experta
**Decisión:** `expert_test_set` = **20 casos difíciles curados por Roger** (skills de skills.sh
especialmente complejas y/o instrucciones trampa redactadas por él), rúbrica manual 0–100. Es
el conjunto con más peso cualitativo.

### 32. Evaluación End-to-End
**Decisión:** la más importante. Blueprint → harness → filesystem en tmpdir → `ast.parse` de scripts
+ validación de SKILL.md → **tasa de skills válidas/ejecutables**. Responde "¿la skill generada
realmente funciona?". Incluye validación contra el spec oficial; opcionalmente se corre
`npx skills-ref validate` sobre la skill materializada (compliance real contra el estándar).

### 33. SkillGen Score (métrica global)
```
SkillGen Score = 0.25·estructural + 0.25·correctitud + 0.20·arquitectura
               + 0.15·completitud + 0.15·ejecución
```
**Justificación de pesos:** sin JSON válido nada sirve (estructural+correctitud = 50%); arquitectura
es la tesis del proyecto (¿elige bien la complejidad?); ejecución cierra el loop.

### 34. Baseline
**Decisión:** modelo base **sin fine-tuning**, mismo system prompt + 2–3 few-shot, mismo test, mismo
juez. Comparación honesta Base vs QLoRA.

### 35. Ablation studies
**Decisión:** una variable por vez: dataset size (200/400/800), LoRA rank (8/16/32), LR (1e-4/2e-4),
epochs (1/2/3), formato de output (con/sin file contents).

### 36. Overfitting
**Decisión:** señal = val↑ + train↓ con brecha >20%, o n-gram overlap train→test en outputs.
Respuesta en orden: early stopping → dropout 0.05 → menos epochs → más datos.

### 37. Catastrophic forgetting
**Decisión:** 50 preguntas generales (subset de ARC-Challenge) antes/después. Caída >5% se documenta
como hallazgo (no bloquea v1).

### 38. Data drift / Retraining
**Decisión:** nueva data → nueva dataset version → **reentrenar desde el base** (no continual
learning, para evitar enredos) → pipeline Accept/Reject (§39).

---

## BLOQUE F — MLOps + deployment (39–49)

### 39. Retraining pipeline
`New data → Validation → Dataset version → Training → Evaluation → Benchmark → Accept/Reject → Model
registry`. **Accept si:** Score ≥ actual + 2 pts **y** sin regresiones en expert set.

### 40. Model Registry
`skillgen-v0` (baseline sin entrenar) → v1, v2… Solo se promueve si supera al anterior en el
benchmark completo.

### 41. Inferencia
**Decisión:** temp=0.3, top_p=0.9, max_new_tokens=4096, repetition_penalty=1.1, stop al cerrar el
JSON. Sin constrained decoding en v1; validación post-hoc + hasta 2 reintentos con el error como
feedback (§42).

### 42. Validación post-inferencia
`JSON parse → schema → validación de skill (paths, seguridad) → accept / repair / reject`.
Repair = reintento con el error como feedback (máx 2). Reject = error tipado al cliente.

### 43. Serving (FastAPI)
`POST /generate-skill` `{input}` → `{blueprint, validation_report}`. v1 local: sin auth, timeout
120s, input máx 8k tokens, concurrencia 1 (cola simple), errores tipados.

### 44. Hugging Face deployment 🟡
**Decisión:** subir **adapters** al Hub (visibilidad la decide Roger en §0.1/pregunta 5). Inferencia:
base 4-bit + adapters en T4 (Kaggle/Colab) o CPU para demo lenta. Cold start = descarga de adapters
(~100–200MB) + carga del base.

### 45. Reproducibilidad
Por experimento: git commit, versiones python/torch/cuda/unsloth/transformers, dataset hash, model
version, config, seed + `reproduce.sh`.

### 46. Random seeds
Seed principal **42**. El experimento final aceptado se repite con seeds 7 y 123 (costo controlado);
variabilidad aceptable ±3 pts en SkillGen Score.

### 47. Failure analysis
Por experimento: `experiments/<id>/failures.md` con BEST / WORST / REGRESSION / NEW FAILURE,
clasificados E1–E10.

### 48. Taxonomía de errores
Se adopta E1–E10 del cuestionario tal cual (E1 invalid structure … E10 format failure).

### 49. Comparación contra modelos externos
Opcional, al final: SkillGen vs base vs Claude en el expert set. Solo para estudiar
**especialización vs capacidad general**, no como objetivo.

---

## BLOQUE G — Repository + implementación (50–58)

### 50. Límites del proyecto
Se adopta la lista del cuestionario: no training desde cero, no modelos gigantes, no RLHF/DPO en v1
(aunque se guardan pares), no ensembles, no agente autónomo completo, no retraining automático real.

### 51. Estructura del repositorio
```
skillgen/
├── notebooks/          00_environment, 01_pytorch, 02_transformers, 03_baseline,
│                       04_lora_experiment, 05_qlora_training, 06_evaluation
├── data/               raw/ processed/ dataset_v1/{train.jsonl, validation.jsonl, test.jsonl}
│                       expert/ CHANGELOG.md
├── src/skillgen/       data/ training/ evaluation/ inference/ serving/ harness/
├── configs/            qlora.yaml, eval.yaml, dataset.yaml
├── scripts/            train.py, evaluate.py, inference.py, build_dataset.py
├── experiments/        <id>/{config.yaml, metrics.json, failures.md, checkpoints/}
├── tests/
├── docs/               DECISIONS.md, FINDINGS.md, ERROR_TAXONOMY.md
└── README.md
```

### 52. Código de entrenamiento
Funciones del cuestionario (`load_model`, `load_tokenizer`, `load_dataset`, `prepare_dataset`,
`configure_quantization`, `configure_lora`, `configure_training`, `train`, `evaluate`,
`save_adapter`). Reparto: notebooks = aprendizaje/exploración; `src/` = módulos reutilizables;
`configs/*.yaml` = hiperparámetros; `scripts/*.py` = CLI delgado.

### 53. Configuración
**YAML** (formato del §18). Python solo para defaults internos.

### 54. CLI
`train.py --config`, `evaluate.py --model`, `inference.py --model`, + `build_dataset.py --config`.

### 55. Tests
pytest: parser JSONL, schema, blueprint validator (incl. paths maliciosos `../`), tokenizer
roundtrip, harness en tmpdir, API con TestClient, funciones de eval con fixtures.

### 56. Seguridad del output
El validator rechaza: `..`, paths absolutos, extensiones fuera de allowlist
(.md/.py/.yaml/.json/.txt), imports peligrosos en scripts (`os.system`, `subprocess`, `socket`,
`eval`, …). En v1 el código generado **no se ejecuta** (solo `ast.parse`); ejecución sandboxed
queda fuera de v1.

### 57. ¿Qué significa "éxito"?
**Δ mínimo significativo: +10 puntos absolutos** en SkillGen Score sobre el baseline few-shot, con
mejora en ≥4/5 dimensiones y sin regresión en expert set. Si no se alcanza, el proyecto igual
cumplió su objetivo de aprendizaje y se documenta por qué (hallazgo, no fracaso).

### 58. Resultado final esperado
El diagrama del cuestionario, demostrable end-to-end: input → blueprint → harness → skill válida →
Score(Base) vs Score(QLoRA), con cada bloque explicado técnicamente.

---

## Orden de ejecución

1. Congelar las 5 decisiones (§0.1) → crear repo `skillgen`.
2. Esqueleto del repo + configs + harness + validator (todo testeable **sin GPU**).
3. Dataset: seleccionar skills de skills.sh (filtro §11) → generar instrucciones sintéticas →
   test+expert primero, luego train (pipeline §11–13).
   **Piloto hecho 2026-09-26:** 73 ejemplos en `data/pilot/dataset_pilot.jsonl`
   (scripts: `select_pilot.py`, `convert_skill.py`, `build_pilot.py`). Pool: 38 repos, ~1.300
   skills. Curación: 10 nombres normalizados al spec, 11 descartadas con razón
   (`data/rejected_pilot.json`: stubs, sin frontmatter, sin licencia). Licencias: 53 MIT +
   20 Apache-2.0 (verificadas; 4 skills sin licencia descartadas). Instrucciones en inglés
   (pendiente confirmar idioma con Roger).
4. Notebooks 00–04 (ambiente, PyTorch, baseline, LoRA educativo).
5. Entrenamiento E0→E1 en Kaggle (T4) + W&B.
6. Evaluación completa + failures.md + SkillGen Score.
7. FastAPI + subida de adapters a HF.
8. Comparación externa opcional + FINDINGS.md.
