"""Catastrophic-forgetting check (PLAN.md §37).

PLAN §37: 50 general questions (ARC-Challenge subset) before/after
fine-tuning; a drop > 5% is documented as a finding (it does not block v1).

Scoring is by **loglikelihood**, not generation: for each question, each
option is scored as the continuation of the question prompt, and the
argmax option is the model's answer. That is fast on CPU and needs no
sampling settings.

Question sources:
- :func:`load_arc_questions` — ARC-Challenge from HF ``datasets`` when
  available (the real §37 run, on Kaggle); labelled ``source="arc"``.
- :data:`LOCAL_QUESTIONS` — 20 **original** multiple-choice questions
  written for this repo. They are **NOT ARC** and must never be cited as
  ARC. They exist so the before/after harness can be smoke-tested on CPU
  with zero downloads beyond the model itself.
"""

from __future__ import annotations

# Original local set — written for SkillGen, NOT ARC. 4 options each.
LOCAL_QUESTIONS = [
    {"question": "Which planet in our solar system has the largest ring system?",
     "options": ["Jupiter", "Saturn", "Uranus", "Neptune"], "answer": 1, "topic": "science"},
    {"question": "What is the chemical symbol for gold?",
     "options": ["Ag", "Fe", "Au", "Gd"], "answer": 2, "topic": "science"},
    {"question": "Which organ in the human body produces insulin?",
     "options": ["Liver", "Pancreas", "Spleen", "Kidney"], "answer": 1, "topic": "science"},
    {"question": "What force keeps planets in orbit around the Sun?",
     "options": ["Magnetism", "Friction", "Gravity", "Inertia"], "answer": 2, "topic": "science"},
    {"question": "In which year did humans first land on the Moon?",
     "options": ["1959", "1965", "1969", "1975"], "answer": 2, "topic": "history"},
    {"question": "Which ancient civilisation built the pyramids of Giza?",
     "options": ["The Romans", "The Aztecs", "The Egyptians", "The Greeks"], "answer": 2, "topic": "history"},
    {"question": "The Berlin Wall fell in which year?",
     "options": ["1979", "1985", "1989", "1991"], "answer": 2, "topic": "history"},
    {"question": "Who was the first President of the United States?",
     "options": ["Thomas Jefferson", "Abraham Lincoln", "John Adams", "George Washington"], "answer": 3, "topic": "history"},
    {"question": "What is 15% of 200?",
     "options": ["20", "25", "30", "45"], "answer": 2, "topic": "math"},
    {"question": "What is the next number in the sequence 2, 6, 12, 20, 30?",
     "options": ["36", "40", "42", "44"], "answer": 2, "topic": "math"},
    {"question": "If a train travels 60 km in 45 minutes, what is its average speed in km/h?",
     "options": ["70", "75", "80", "90"], "answer": 2, "topic": "math"},
    {"question": "What is the square root of 144?",
     "options": ["12", "14", "16", "24"], "answer": 0, "topic": "math"},
    {"question": "Which is the longest river in South America?",
     "options": ["Parana River", "Amazon River", "Orinoco River", "Sao Francisco River"], "answer": 1, "topic": "geography"},
    {"question": "What is the capital of Australia?",
     "options": ["Sydney", "Melbourne", "Canberra", "Perth"], "answer": 2, "topic": "geography"},
    {"question": "Which desert is the largest hot desert in the world?",
     "options": ["Gobi", "Sahara", "Kalahari", "Mojave"], "answer": 1, "topic": "geography"},
    {"question": "Mount Kilimanjaro is located in which country?",
     "options": ["Kenya", "Ethiopia", "Tanzania", "Uganda"], "answer": 2, "topic": "geography"},
    {"question": "Which word is a synonym of 'rapid'?",
     "options": ["Slow", "Swift", "Heavy", "Quiet"], "answer": 1, "topic": "language"},
    {"question": "In English grammar, which of these is a verb?",
     "options": ["Beautiful", "Quickly", "Run", "Happiness"], "answer": 2, "topic": "language"},
    {"question": "What is the plural of 'octopus' in standard usage?",
     "options": ["Octopi", "Octopuses", "Octopodes only", "Octopus"], "answer": 1, "topic": "language"},
    {"question": "Which sentence is grammatically correct?",
     "options": ["She don't like coffee.", "She doesn't likes coffee.",
                 "She doesn't like coffee.", "She not like coffee."], "answer": 2, "topic": "language"},
]


def load_arc_questions(limit: int = 50) -> tuple[list[dict], str]:
    """Load ARC-Challenge questions from HF datasets if available.

    Returns (questions, source) where source is "arc" if the real
    ARC-Challenge test split was loaded, otherwise "local" with the
    original 20-question fallback set.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        questions = []
        for row in ds:
            texts = row["choices"]["text"]
            labels = row["choices"]["label"]
            if row["answerKey"] not in labels or len(texts) < 2:
                continue
            questions.append({
                "question": row["question"], "options": list(texts),
                "answer": labels.index(row["answerKey"]), "topic": "arc",
            })
            if len(questions) >= limit:
                break
        if questions:
            return questions, "arc"
    except Exception:
        pass
    return [dict(q) for q in LOCAL_QUESTIONS], "local"


def score_questions(model, tokenizer, questions: list[dict]) -> list[dict]:
    """Answer each question by argmax loglikelihood over its options.

    Each option is scored as the continuation of
    ``"Question: ...\\nAnswer:"``; the score is the sum log-probability of
    the option tokens (the standard ARC-style choice scoring).
    """
    import torch

    results = []
    for q in questions:
        prompt = f"Question: {q['question']}\nAnswer:"
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        scores = []
        for option in q["options"]:
            full = tokenizer(prompt + " " + option, return_tensors="pt")["input_ids"]
            with torch.no_grad():
                logits = model(full).logits
            logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
            target = full[0, 1:]
            token_lp = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            scores.append(float(token_lp[prompt_ids.shape[-1] - 1:].sum()))
        predicted = int(max(range(len(scores)), key=lambda i: scores[i]))
        results.append({
            "question": q["question"], "options": q["options"],
            "answer": q["answer"], "predicted": predicted,
            "correct": predicted == q["answer"], "topic": q.get("topic", ""),
        })
    return results


def run_forgetting_check(model_id: str, adapter_path: str | None = None,
                         questions: list[dict] | None = None) -> dict:
    """Run the forgetting check for a base model (+ optional LoRA adapter).

    Returns {"accuracy": float, "n": int, "per_question": [...]}.
    Heavy imports are local so the module stays importable without torch.
    """
    import torch  # noqa: F401  (ensures a clear error if torch is missing)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if questions is None:
        questions, _source = load_arc_questions()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    per_question = score_questions(model, tokenizer, questions)
    n = len(per_question)
    correct = sum(1 for r in per_question if r["correct"])
    return {"model_id": model_id, "adapter_path": adapter_path,
            "n": n, "correct": correct, "accuracy": correct / n if n else 0.0,
            "per_question": per_question}
