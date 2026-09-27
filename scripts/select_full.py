#!/usr/bin/env python3
"""Full-dataset selection: stratified sample of N skills across domains (fixed seed).

Excludes the pilot ids, filters repos/skills by permissive license, dedups
translated/canonical copies. Usage:
    python scripts/select_full.py --n 927 --seed 42 --out data/selection_full.jsonl
"""
import argparse
import json
import random
import re
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"

# --- license check (mirrors convert_skill.py) ---
PERMISSIVE = ("mit", "apache-2.0", "bsd", "cc0", "unlicense")


def guess_license(base: Path):
    for cand in ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "LICENCE.md"):
        if (base / cand).exists():
            text = (base / cand).read_text(errors="replace")[:400].lower()
            if "apache license" in text and "version 2.0" in text:
                return "apache-2.0"
            for lic in ("mit", "bsd", "cc0", "unlicense"):
                if lic in text:
                    return lic
            return "present (unparsed)"
    return "unknown"


def effective_license(repo_root: Path, skill_dir: Path, cache: dict):
    """Skill-level LICENSE wins; else repo-level. Only permissive passes."""
    lic = guess_license(skill_dir)
    if lic in PERMISSIVE:
        return lic
    if repo_root not in cache:
        cache[repo_root] = guess_license(repo_root)
    repo_lic = cache[repo_root]
    if lic in PERMISSIVE:
        return lic
    if repo_lic in PERMISSIVE:
        return repo_lic
    return None  # not usable


# --- deprecation-stub filter ---
def is_stub_skill(skill_dir: Path) -> bool:
    """Deprecated signpost stubs (moved/redirect) are not real training targets."""
    md = skill_dir / "SKILL.md"
    if not md.exists():
        return True
    sm = md.read_text(errors="replace").lower()
    m = re.search(r"^---\s*\n(.*?)\n---", sm, re.S)
    desc = m.group(1) if m else ""
    return (re.search(r"status:\s*moved", sm) is not None
            or "signpost" in sm
            or "deprecated compatibility entry" in desc
            or "no longer developed here" in sm)


# --- classification ---
REPO_DOMAIN = {
    "kostja94_marketing-skills": "marketing",
    "coreyhaines31_marketingskills": "marketing",
    "appeeky_aso-skills": "marketing",
    "agricidaniel_claude-seo": "marketing",
    "aaron-he-zhu_seo-geo-claude-skills": "marketing",
    "axelfreeman_marketing-mindset": "marketing",
    "joellewis_finance_skills": "finance",
    "himself65_finance-skills": "finance",
    "stanleychanh_tushare-finance-skill-for-claude-code": "finance",
    "nexscope-ai_ecommerce-skills": "sales",
    "refoundai_lenny-skills": "business",
    "phuryn_pm-skills": "business",
    "riekelt_technical-writer": "writing",
    "miqdadbadjuber_anti-slop": "writing",
    "vercel-labs_agent-browser": "software",
    "vercel-labs_agent-skills": "software",
    "anthropics_skills": None,  # mixed: classify by name
    "anthropics_knowledge-work-plugins": None,  # mixed: kwp categories
    "claude-office-skills_skills": "productivity",
    "affaan-m_ecc": None,  # mixed: classify by name
    "googleworkspace_cli": "productivity",
    "larksuite_cli": "productivity",
    "get-dx_cli": "software",
    "inference-sh_skills": "software",
    "supabase_agent-skills": "software",
    "stripe_link-cli": "software",
    "cloudflare_skills": "software",
    "prisma_skills": "software",
    "langchain-ai_langchain-skills": "software",
    "shopify_shopify-ai-toolkit": "sales",
    "forcedotcom_sf-skills": "sales",
    "wshobson_agents": "software",
    "mattpocock_skills": "software",
    "emilkowalski_skills": "software",
    "brave_brave-search-skills": "research",
    "tavily-ai_skills": "research",
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
}

NAME_HINTS = [
    (r"ads\b|seo|brand|social|growth|funnel|campaign|tiktok|influencer|content", "marketing"),
    (r"lead|prospect|crm|quota|shopify|ecommerce|ebay|etsy|walmart|amazon|checkout", "sales"),
    (r"financ|account|tax|invoice|billing|crypto|trading|backtest|trade|budget|expense|payroll", "finance"),
    (r"support|ticket|escalat|helpdesk", "support"),
    (r"operations|\bops\b|oncall|incident|vendor|procurement|logistics|supply", "operations"),
    (r"legal|contract|nda|compliance|hiring|interview|onboard|hr\b|recruit|leadership|manag", "business"),
    (r"research|digest|survey|interview", "research"),
    (r"\bdata\b|dashboard|analytics|visual|sql|statistic|report|scrap|metric", "analytics"),
    (r"meeting|productiv|assistant|email|calendar|notes|slides|pdf|template|notion|workflow", "productivity"),
    (r"kotlin|golang|django|backend|pattern|cache|testing|perl|rust|coroutine|android|ios|mobile|code|deploy|debug|\bapi\b|sdk|cli\b|mcp\b|devops|docker|k8s", "software"),
    (r"writ|copy|editor|author|blog|postmortem|runbook|poem|story|novel", "writing"),
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
    dom = REPO_DOMAIN.get(repo)
    if dom:
        return dom
    for pat, d in NAME_HINTS:
        if re.search(pat, skill_name):
            return d
    return "business"


QUOTAS = {
    "software": 138, "marketing": 112, "productivity": 109, "finance": 91,
    "sales": 83, "writing": 75, "research": 75, "support": 65,
    "analytics": 65, "operations": 56, "business": 58,
}  # = 927; plus 73 pilot = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(BASE / "data" / "selection_full.jsonl"))
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="extra selection .jsonl files whose ids to exclude")
    ap.add_argument("--quotas", default=None,
                    help='JSON dict overriding QUOTAS, e.g. \'{"writing": 60}\'')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pilot_ids = {json.loads(l)["id"] for l in open(BASE / "data" / "pilot_selection.jsonl")}
    for xf in args.exclude:
        pilot_ids.update(json.loads(l)["id"] for l in open(xf))
    quotas = QUOTAS if args.quotas is None else json.loads(args.quotas)
    lic_cache, skipped_no_license = {}, Counter()

    by_key = {}
    repos_root = RAW / "_repos"
    for skill_md in sorted(repos_root.rglob("SKILL.md")):
        if ".git" in skill_md.parts:
            continue
        rel_to_repos = skill_md.parent.relative_to(repos_root)
        parts = rel_to_repos.parts
        repo, rel = parts[0], "/".join(parts[1:])
        skill_name = parts[-1]
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
        sid = f"{repo}__{skill_name}"
        if sid in pilot_ids:
            continue
        repo_root = repos_root / repo
        skill_dir = repos_root / repo / rel
        if is_stub_skill(skill_dir):
            skipped_no_license[f"stub:{repo}"] += 1
            continue
        lic = effective_license(repo_root, skill_dir, lic_cache)
        if not lic:
            skipped_no_license[repo] += 1
            continue
        skills.append({
            "id": sid, "repo": repo, "relpath": rel, "skill_name": skill_name,
            "domain": classify(repo, rel, skill_name), "license": lic,
        })

    print(f"pool: {len(skills)} licensed skills "
          f"(excluded {sum(skipped_no_license.values())} without permissive license)")
    by_domain = {}
    for s in skills:
        by_domain.setdefault(s["domain"], []).append(s)
    for d in by_domain:
        rng.shuffle(by_domain[d])

    selected = []
    short = {}
    for domain, quota in quotas.items():
        pool = by_domain.get(domain, [])
        take = min(quota, len(pool))
        selected.extend(pool[:take])
        if take < quota:
            short[domain] = (len(pool), quota)
    rng.shuffle(selected)

    out = Path(args.out)
    with open(out, "w") as f:
        for s in selected:
            f.write(json.dumps(s) + "\n")
    print(f"selected {len(selected)} -> {out}")
    print(Counter(s["domain"] for s in selected))
    if short:
        print("SHORTFALLS:", short)


if __name__ == "__main__":
    main()
