#!/usr/bin/env python3
"""Stratified pilot selection of skills across domains (fixed seed).

Usage:
    python scripts/select_pilot.py [--n 80] [--seed 42] [--out data/pilot_selection.jsonl]
"""
import argparse
import json
import random
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"

REPO_DOMAIN = {
    "kostja94_marketing-skills": "marketing",
    "wind-alice_alicemarket": "finance",  # trading-focused repo
    "coreyhaines31_marketingskills": "marketing",
    "appeeky_aso-skills": "marketing",
    "agricidaniel_claude-seo": "marketing",
    "aaron-he-zhu_seo-geo-claude-skills": "marketing",
    "axelfreeman_marketing-mindset": "marketing",
    "joellewis_finance_skills": "finance",
    "himself65_finance-skills": "finance",
    "binance_binance-skills-hub": "finance",
    "stanleychanh_tushare-finance-skill-for-claude-code": "finance",
    "nexscope-ai_ecommerce-skills": "sales",
    "refoundai_lenny-skills": "business",
    "phuryn_pm-skills": "business",
    "riekelt_technical-writer": "writing",
    "miqdadbadjuber_anti-slop": "writing",
    "vercel-labs_agent-browser": "software",
    "vercel-labs_agent-skills": "software",
}

# Repos with mixed contents: classify by skill name, fallback to default.
MIXED_REPOS = {
    "claude-office-skills_skills": "productivity",
    "affaan-m_ecc": None,
}

KWP_CATEGORY_DOMAIN = {
    "bio-research": "research",
    "customer-support": "support",
    "data": "analytics",
    "design": "business",
    "engineering": "software",
    "enterprise-search": "research",
    "finance": "finance",
    "human-resources": "business",
    "legal": "business",
    "marketing": "marketing",
    "operations": "operations",
    "product-management": "business",
    "productivity": "productivity",
    "sales": "sales",
    "small-business": "business",
    "cowork-plugin-management": "software",
    "partner-built": None,  # decide by name below
}

QUOTAS = {
    "marketing": 12, "finance": 9, "sales": 7, "support": 6, "operations": 5,
    "research": 5, "analytics": 7, "business": 10, "writing": 6,
    "productivity": 8, "software": 5,
}

NAME_HINTS = [
    (r"ads|seo|brand|social|growth|funnel|campaign", "marketing"),
    (r"lead|prospect|crm|quota|woocommerce|shopify|ecommerce|ebay|etsy", "sales"),
    (r"financ|account|tax|invoice|billing|crypto|trading|backtest|trade", "finance"),
    (r"support|ticket|escalat", "support"),
    (r"audit-ops|operations|\bops\b|oncall|incident|vendor|procurement", "operations"),
    (r"legal|contract|nda|compliance|job|hiring|interview|onboard|performance", "business"),
    (r"research|digest", "research"),
    (r"data|dashboard|analytics|visual|sql|statistic|report|scrap", "analytics"),
    (r"meeting|productiv|assistant|email|calendar|notes|slides|pdf|template", "productivity"),
    (r"kotlin|golang|django|backend|pattern|cache|testing|perl|rust|coroutine|android|ios|mobile|code|deploy|debug|\bapi\b|sdk", "software"),
    (r"writ|copy|editor|author|blog|postmortem|runbook", "writing"),
]


def classify(repo, relpath, skill_name):
    if repo == "anthropics_knowledge-work-plugins":
        parts = [p for p in relpath.split("/") if p]
        cat = parts[0] if parts else ""
        dom = KWP_CATEGORY_DOMAIN.get(cat)
        if dom:
            if cat == "design" and "copy" in skill_name:
                return "writing"
            return dom
    if repo in MIXED_REPOS:
        for pat, d in NAME_HINTS:
            if re.search(pat, skill_name):
                return d
        return MIXED_REPOS[repo] or "business"
    dom = REPO_DOMAIN.get(repo)
    if dom:
        return dom
    for pat, d in NAME_HINTS:
        if re.search(pat, skill_name):
            return d
    return "business"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(BASE / "data" / "pilot_selection.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_key = {}
    repos_root = RAW / "_repos"
    for skill_md in sorted(repos_root.rglob("SKILL.md")):
        if ".git" in skill_md.parts:
            continue
        rel_to_repos = skill_md.parent.relative_to(repos_root)
        parts = rel_to_repos.parts
        repo, rel = parts[0], "/".join(parts[1:])
        skill_name = parts[-1]
        # Prefer canonical copy: skills/ > .agents/skills/ > other > docs/<lang> translations
        if rel == "skills" or rel.startswith("skills/"):
            score = 0
        elif ".agents/skills" in rel:
            score = 1
        elif rel.startswith("docs/"):
            score = 3
        else:
            score = 2
        key = (repo, skill_name)
        if key not in by_key or score < by_key[key][0]:
            by_key[key] = (score, repo, rel, skill_name)
    skills = []
    for score, repo, rel, skill_name in sorted(by_key.values()):
        skills.append({
            "id": f"{repo}__{skill_name}",
            "repo": repo,
            "relpath": rel,
            "skill_name": skill_name,
            "domain": classify(repo, rel, skill_name),
        })

    by_domain = {}
    for s in skills:
        by_domain.setdefault(s["domain"], []).append(s)
    for d in by_domain:
        rng.shuffle(by_domain[d])

    selected = []
    for domain, quota in QUOTAS.items():
        pool = by_domain.get(domain, [])
        take = min(quota, len(pool))
        selected.extend(pool[:take])
        if take < quota:
            print(f"WARN: domain {domain}: only {len(pool)} available, wanted {quota}")
    rng.shuffle(selected)

    out = Path(args.out)
    with open(out, "w") as f:
        for s in selected:
            f.write(json.dumps(s) + "\n")
    print(f"selected {len(selected)} skills -> {out}")
    from collections import Counter
    print(Counter(s["domain"] for s in selected))


if __name__ == "__main__":
    main()
