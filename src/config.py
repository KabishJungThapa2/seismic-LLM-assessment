# =============================================================================
# config.py — Building era defaults, AS1170.4 constants, Claude LLM config
# seismic-llm-assessment | UTS Engineering Graduate Project PG (42003)
#
# LLM CHOICE: Claude (Anthropic) over GPT-4o (OpenAI)
# Full rationale: docs/LLM_CHOICE.md
#
# Summary of why Claude was chosen:
# 1. Constitutional AI training → more honest about uncertainty
# 2. Explicitly flags assumptions (critical for safety-critical assessment)
# 3. Claude Haiku is ~6-10x cheaper than GPT-4o per extraction
# 4. Research novelty — Liang et al. (2025a) used GPT-4o; using Claude
#    provides independent comparative data point
# 5. Consistent structured JSON output without markdown contamination
# =============================================================================

CLAUDE_MODEL_PRIMARY = "claude-sonnet-4-6"
CLAUDE_MODEL_FAST    = "claude-haiku-4-5-20251001"

ERA_DEFAULTS = {
    "pre-1990": {
        "fc":20.0,"fy":250.0,"col_b":0.30,"col_h":0.30,
        "beam_b":0.30,"beam_h":0.45,"col_rho":0.015,
        "beam_rho_t":0.008,"beam_rho_c":0.004,"mu":2.0,"Sp":0.77,
        "col_mod":0.50,"beam_mod":0.35,"epsc0_core":-0.004,"epsU_core":-0.012,
    },
    "post-1990": {
        "fc":32.0,"fy":500.0,"col_b":0.35,"col_h":0.35,
        "beam_b":0.30,"beam_h":0.50,"col_rho":0.020,
        "beam_rho_t":0.012,"beam_rho_c":0.006,"mu":3.0,"Sp":0.67,
        "col_mod":0.60,"beam_mod":0.40,"epsc0_core":-0.005,"epsU_core":-0.020,
    },
    "post-2010": {
        "fc":40.0,"fy":500.0,"col_b":0.40,"col_h":0.40,
        "beam_b":0.35,"beam_h":0.55,"col_rho":0.025,
        "beam_rho_t":0.015,"beam_rho_c":0.0075,"mu":4.0,"Sp":0.67,
        "col_mod":0.70,"beam_mod":0.50,"epsc0_core":-0.006,"epsU_core":-0.030,
    },
}

HAZARD_FACTORS = {
    "newcastle":0.11,"sydney":0.08,"melbourne":0.08,"brisbane":0.05,
    "adelaide":0.10,"perth":0.09,"canberra":0.08,"hobart":0.05,
    "darwin":0.09,"default":0.11,
}

def spectral_shape_De(T1):
    if T1 <= 0.10: return 2.35
    elif T1 < 1.50: return 1.65 * (0.1/T1)**0.85
    else: return 1.10 * (1.5/T1)**2.0

def spectral_shape_Ce(T1):
    if T1 <= 0.10: return 2.35
    elif T1 < 1.50: return 1.35 * (0.1/T1)**0.80
    else: return 0.90 * (1.5/T1)**2.0

SPECTRAL_SHAPE  = {"De": spectral_shape_De, "Ce": spectral_shape_Ce}
DRIFT_LIMIT     = 0.015
MIN_BASE_SHEAR  = 0.01
COVER           = 0.040
G               = 9.81

PARAM_BOUNDS = {
    "num_storeys":(1,4),"storey_height":(2.4,4.5),"num_bays":(1,6),
    "bay_width":(2.5,8.0),"fc":(15.0,65.0),"fy":(200.0,600.0),
    "col_b":(0.20,0.80),"col_h":(0.20,0.80),"beam_b":(0.20,0.60),
    "beam_h":(0.25,0.80),"col_rho":(0.01,0.04),"mu":(1.5,6.0),"Z":(0.03,0.45),
}

LLM_SYSTEM_PROMPT = """You are a structural engineering expert specialising in
Australian residential building seismic assessment under AS1170.4.

Extract structural parameters from the building description and return ONLY
a valid JSON object. No markdown fences, no preamble, no explanation.

Required JSON fields:
{
  "building_name": "short descriptive name",
  "num_storeys": integer 1-4,
  "storey_height": float metres (default 3.0),
  "num_bays": integer 2-5 (default 3),
  "bay_width": float metres (default 4.0),
  "floor_width": float metres (building width, default 8.0),
  "fc": float MPa (20 pre-1990 / 32 post-1990 / 40 post-2010),
  "fy": float MPa (250 pre-1990 / 500 post-1990 or post-2010),
  "col_b": float metres (0.30/0.35/0.40 by era),
  "col_h": float metres (same as col_b for square),
  "beam_b": float metres (0.30 default),
  "beam_h": float metres (0.45/0.50/0.55 by era),
  "col_rho": float (0.015/0.020/0.025 by era),
  "beam_rho_t": float (0.008/0.012/0.015 by era),
  "beam_rho_c": float (half of beam_rho_t),
  "mu": float (2.0/3.0/4.0 by era),
  "Sp": float (0.77 for mu=2.0, else 0.67),
  "Z": float (0.11 Newcastle-Sydney / 0.08 Melbourne / 0.05 Brisbane),
  "site_class": "De or Ce (default De for coastal NSW)",
  "dead_load": float kPa (default 5.0),
  "live_load": float kPa (default 2.0 residential),
  "era": "pre-1990 or post-1990 or post-2010",
  "confidence": float 0.0-1.0,
  "assumptions": ["every assumption made — be exhaustive, this is safety-critical"]
}

Rules: pre-1990=no seismic code poor confinement. post-1990=AS1170.4:1993 moderate.
post-2010=AS1170.4:2007 full ductility. Default Z=0.11 Newcastle if unclear.
'old/heritage/brick veneer'→pre-1990. 'modern/new'→post-2010."""
