# =============================================================================
# extractor.py — LLM parameter extraction using Claude (Anthropic)
# seismic-llm-assessment | UTS Engineering Graduate Project PG (42003)
#
# WHY CLAUDE OVER GPT-4o:
#   Constitutional AI training makes Claude more honest about uncertainty —
#   it explicitly flags assumptions in output rather than filling gaps
#   silently. This is critical for safety-critical seismic assessment.
#   Claude Haiku is also ~6-10x cheaper than GPT-4o per extraction, directly
#   supporting the project's democratisation objective.
#   Full rationale: docs/LLM_CHOICE.md
# =============================================================================

import json, re, os
from config import (ERA_DEFAULTS, HAZARD_FACTORS, LLM_SYSTEM_PROMPT,
                    PARAM_BOUNDS, CLAUDE_MODEL_PRIMARY, CLAUDE_MODEL_FAST)


def extract(description: str, api_key: str = None,
            fast_mode: bool = False) -> dict:
    """
    Extract structural parameters from a natural language building description.

    Args:
        description: Plain English building description
        api_key:     Anthropic API key. If None checks ANTHROPIC_API_KEY env var.
        fast_mode:   Use claude-haiku (faster/cheaper) instead of claude-sonnet.

    Returns:
        dict with structural parameters, confidence score, and assumptions list.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    if key:
        model = CLAUDE_MODEL_FAST if fast_mode else CLAUDE_MODEL_PRIMARY
        print(f"  Using Claude ({model})...")
        params = _claude_extract(description, key, model)
        if params:
            print(f"  ✓ Claude extraction complete "
                  f"(confidence: {params.get('confidence', 0):.0%})")
            return params
        print("  Claude API failed — falling back to demo mode")

    print("  Demo mode (keyword-based, no API key needed)")
    return _demo_extract(description)


def _claude_extract(description: str, api_key: str, model: str) -> dict | None:
    """Call Anthropic Claude API and parse JSON response."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            system=LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": description}],
        )

        raw = msg.content[0].text.strip()
        # Strip any accidental markdown fences
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$',           '', raw, flags=re.MULTILINE)
        raw = raw.strip()

        return json.loads(raw)

    except ImportError:
        print("  anthropic package not installed — run: pip install anthropic")
        return None
    except json.JSONDecodeError as e:
        print(f"  Could not parse Claude response as JSON: {e}")
        return None
    except Exception as e:
        print(f"  Claude API error: {e}")
        return None


def _demo_extract(description: str) -> dict:
    """
    Keyword-based extraction that simulates Claude behaviour.
    No API key required. Covers three construction eras, Australian cities,
    and common floor plan description patterns.
    """
    d           = description.lower()
    assumptions = []

    era         = _detect_era(d, assumptions)
    props       = ERA_DEFAULTS[era].copy()
    assumptions.append(f"Material/section properties from {era} era defaults")

    num_storeys                         = _detect_storeys(d, assumptions)
    floor_length, floor_width, num_bays, bay_width = _detect_geometry(d, assumptions)
    Z, site_class                       = _detect_zone(d, assumptions)

    return {
        "building_name":  f"{era.capitalize()} RC Frame ({num_storeys}-storey)",
        "num_storeys":    num_storeys,
        "storey_height":  3.0,
        "num_bays":       num_bays,
        "bay_width":      round(bay_width, 2),
        "floor_width":    floor_width,
        "fc":             props["fc"],
        "fy":             props["fy"],
        "col_b":          props["col_b"],
        "col_h":          props["col_h"],
        "beam_b":         props["beam_b"],
        "beam_h":         props["beam_h"],
        "col_rho":        props["col_rho"],
        "beam_rho_t":     props["beam_rho_t"],
        "beam_rho_c":     props["beam_rho_c"],
        "mu":             props["mu"],
        "Sp":             props["Sp"],
        "Z":              Z,
        "site_class":     site_class,
        "dead_load":      5.0,
        "live_load":      2.0,
        "era":            era,
        "confidence":     0.70,
        "assumptions":    assumptions,
    }


def _detect_era(d: str, assumptions: list) -> str:
    pre_kws  = [str(y) for y in range(1960,1990)] + [
        'pre-1990','pre 1990','old','heritage','original',
        'unreinforced','brick veneer','non-ductile','fibrous cement','asbestos']
    post10   = [str(y) for y in range(2011,2026)] + [
        'post-2010','post 2010','modern','new build','newly built',
        'recently built','high strength','fully ductile','contemporary']
    post90   = [str(y) for y in range(1990,2011)] + ['post-1990','post 1990']

    if any(k in d for k in pre_kws):   return "pre-1990"
    if any(k in d for k in post10):    return "post-2010"
    if any(k in d for k in post90):    return "post-1990"
    assumptions.append("Era not specified — defaulted to pre-1990 (conservative)")
    return "pre-1990"


def _detect_storeys(d: str, assumptions: list) -> int:
    mapping = {
        1: ['one storey','1 storey','1-storey','single storey','bungalow'],
        2: ['two storey','2 storey','2-storey','two-storey','double storey'],
        3: ['three storey','3 storey','3-storey','three-storey'],
        4: ['four storey','4 storey','4-storey'],
    }
    for n, kws in mapping.items():
        if any(k in d for k in kws): return n
    import re
    m = re.search(r'(\d)\s*(?:floor|level)', d)
    if m: return int(m.group(1))
    if 'storey' not in d and 'story' not in d and 'floor' not in d:
        assumptions.append("Storeys not specified — defaulted to 2")
    return 2


def _detect_geometry(d: str, assumptions: list):
    import re
    for pat in [r'(\d+(?:\.\d+)?)\s*(?:m|metres?)?\s*(?:x|by|×)\s*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*×\s*(\d+(?:\.\d+)?)']:
        m = re.search(pat, d)
        if m:
            l, w = float(m.group(1)), float(m.group(2))
            if w > l: l, w = w, l
            nb = max(2, round(l/4))
            return l, w, nb, l/nb
    assumptions.append("Floor plan not stated — defaulted to 12m × 8m, 3 bays × 4.0m")
    return 12.0, 8.0, 3, 4.0


def _detect_zone(d: str, assumptions: list):
    city_map = {
        'newcastle':('newcastle','De'),'sydney':('sydney','De'),
        'nsw':('sydney','De'),'wollongong':('sydney','De'),
        'melbourne':('melbourne','Ce'),'victoria':('melbourne','Ce'),
        'brisbane':('brisbane','Ce'),'queensland':('brisbane','Ce'),
        'adelaide':('adelaide','Ce'),'perth':('perth','Ce'),
        'western australia':('perth','Ce'),'canberra':('canberra','Ce'),
        'hobart':('hobart','Ce'),'darwin':('darwin','Ce'),
    }
    for kw,(city,site) in city_map.items():
        if kw in d: return HAZARD_FACTORS[city], site
    assumptions.append("Location not identified — defaulted to Newcastle Z=0.11, Site De")
    return HAZARD_FACTORS['default'], 'De'


def validate(params: dict) -> list:
    """Return list of warning strings for out-of-range parameters."""
    warnings = []
    for key, (lo, hi) in PARAM_BOUNDS.items():
        val = params.get(key)
        if val is None:
            warnings.append(f"MISSING: {key}")
            continue
        if not (lo <= float(val) <= hi):
            warnings.append(f"OUT OF RANGE: {key}={val}  (expected {lo}–{hi})")
    if params.get('beam_h',0) <= params.get('beam_b',999):
        warnings.append("GEOMETRY: beam depth should exceed beam width")
    return warnings
