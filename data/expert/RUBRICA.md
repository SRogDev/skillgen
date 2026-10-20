# Rúbrica manual del expert set (PLAN §31)

0–100 por caso. Se evalúa **a mano** (rúbrica de ~5 min/ejemplo, PLAN §28):
corre el candidato sobre la instrucción, materializa con el harness y puntúa.

| Dimensión | Puntos | Qué mide |
|---|---|---|
| Validez estructural | 20 | JSON parseable, schema del blueprint válido (L1+L2 del harness). |
| Fidelidad al pedido | 20 | Cubre lo pedido; nada inventado fuera del schema; no omite requisitos. |
| Consistencia interna | 20 | name/description/files concuerdan; nombre en kebab-case. |
| Manejo de la dificultad/trampa | 20 | No alucina campos, no obedece instrucciones inválidas ciegamente: normaliza, pide lo mínimo o rechaza con criterio defendible. |
| Ejecutabilidad | 20 | La skill materializada funciona (L5: `ast.parse` de scripts, SKILL.md válido, sin escapes del sandbox). |

**Regla de veto (§39):** un candidato con regresión en este conjunto se
rechaza aunque suba el SkillGen Score. Reporta por caso: promedio,
distribución y la lista de regresiones vs el mejor candidato anterior.
