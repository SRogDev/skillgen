# Expert set (PLAN §31)

20 casos difíciles **curados por Roger**: skills de skills.sh especialmente
complejas y/o instrucciones trampa redactadas por él. Es el conjunto con más
peso cualitativo del proyecto — la puerta §39 (`experiments/README.md`) lo usa
como **veto**: un candidato se rechaza si muestra regresión aquí, aunque suba
el SkillGen Score.

Vive en el conjunto **reservado**: nunca en train, nunca se entrena con ellos,
nunca se modifican una vez congelados.

## Cómo curar cada caso (~10 min)

1. Elige una skill de skills.sh especialmente compleja (muchos archivos,
   referencias cruzadas, instrucciones largas) o redacta una instrucción trampa.
2. Copia `_template.E000.json` a `E001.json` (…`E020.json`) y rellena los campos.
   **No inventes casos por rellenar**: cada uno debe ser un caso que puedas
   defender en una entrevista — por eso los curas tú, no un script.
3. En `difficulty` explica en 1–3 frases por qué es difícil. En
   `curation_notes` anota qué debe contener una buena respuesta — eso alimenta
   tu rúbrica manual cuando evalúes candidatos.
4. Valida el formato: `python3 scripts/validate_expert_set.py`
   (o `pytest tests/test_expert_set.py -v`). Los tests avisan —no fallan— si
   aún faltan casos.

## Ideas de trampas (ligadas a la taxonomía E1–E10, `docs/ERROR_TAXONOMY.md`)

- `nonexistent-field`: pide campos que no existen en el schema. El smoke de la
  Fase 2 (E2/E4 ×5) demostró que el modelo base inventa el esquema; el bueno se
  ciñe al real.
- `contradictory`: instrucciones que se contradicen (p. ej. "sin description"
  cuando el schema la exige).
- `invalid-name`: pide un nombre inválido explícito ("llámala Mi Skill Genial")
  — el bueno normaliza a kebab-case, no obedece ciegamente.
- `sandbox-escape`: pide archivos fuera del directorio de la skill.
- `scope-creep`: un pedido gigante que ninguna skill razonable cubre.
- `ambiguity`: una instrucción que admite dos skills distintas — el bueno elige
  una y es consistente.
- `language-mix`: instrucción en español pidiendo artefactos en inglés (o
  viceversa) — consistencia de idioma.

`trap_type: null` para casos difíciles sin trampa (complejidad pura).

## Rúbrica

Manual, 0–100 por caso: `RUBRICA.md` (5 dimensiones × 20 puntos).
