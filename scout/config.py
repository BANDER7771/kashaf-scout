"""Shared constants: paths, data sources, league strength, positions and attributes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TM_DIR = DATA_DIR / "tm"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = DATA_DIR / "models"

# Free Transfermarkt dataset (github.com/dcaribou/transfermarkt-datasets) - personal/testing use.
TM_BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/"
TM_FILES = ["players", "player_valuations", "transfers", "competitions", "clubs", "appearances"]

UNDERSTAT_URL = "https://understat.com"
# Transfermarkt competition id -> Understat league slug
UNDERSTAT_LEAGUES = {
    "GB1": "EPL", "ES1": "La_liga", "IT1": "Serie_A", "L1": "Bundesliga", "FR1": "Ligue_1", "RU1": "RFPL",
}

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"

# Rough league-strength coefficients (1.0 = Premier League). Used to adjust per-90 output
# across leagues and as a valuation feature. Hand-tuned approximations, not an official ranking.
LEAGUE_STRENGTH = {
    "GB1": 1.00, "ES1": 0.95, "IT1": 0.93, "L1": 0.92, "FR1": 0.88,
    "PO1": 0.80, "NL1": 0.80, "BRA1": 0.78, "BE1": 0.77, "TR1": 0.74,
    "SA1": 0.71, "MLS1": 0.70, "MEX1": 0.70, "ARG1": 0.72, "RU1": 0.72,
    "SC1": 0.68, "A1": 0.70, "C1": 0.68, "DK1": 0.68, "GR1": 0.67,
    "KR1": 0.65, "TS1": 0.66, "PL1": 0.64, "SER1": 0.62, "UKR1": 0.64,
    "NO1": 0.63, "SE1": 0.62, "RO1": 0.60, "JAP1": 0.66, "RSK1": 0.62, "AUS1": 0.58,
    "GB2": 0.74, "QSL": 0.56, "UAE1": 0.56,
}
DEFAULT_STRENGTH = 0.60

# Leagues whose season runs inside one calendar year.
CALENDAR_LEAGUES = {"MLS1", "BRA1", "JAP1", "RSK1", "NO1", "SE1", "ARG1"}

LEAGUE_NAMES_AR = {
    "GB1": "الدوري الإنجليزي", "ES1": "الدوري الإسباني", "IT1": "الدوري الإيطالي",
    "L1": "الدوري الألماني", "FR1": "الدوري الفرنسي", "SA1": "دوري روشن السعودي",
    "PO1": "الدوري البرتغالي", "NL1": "الدوري الهولندي", "TR1": "الدوري التركي",
    "BE1": "الدوري البلجيكي", "BRA1": "الدوري البرازيلي", "MLS1": "الدوري الأمريكي",
    "MEX1": "الدوري المكسيكي", "ARG1": "الدوري الأرجنتيني", "RU1": "الدوري الروسي",
    "SC1": "الدوري الاسكتلندي", "A1": "الدوري النمساوي", "C1": "الدوري السويسري",
    "DK1": "الدوري الدنماركي", "GR1": "الدوري اليوناني", "KR1": "الدوري الكرواتي",
    "TS1": "الدوري التشيكي", "PL1": "الدوري البولندي", "SER1": "الدوري الصربي",
    "UKR1": "الدوري الأوكراني", "NO1": "الدوري النرويجي", "SE1": "الدوري السويدي",
    "RO1": "الدوري الروماني", "JAP1": "الدوري الياباني", "RSK1": "الدوري الكوري",
    "AUS1": "الدوري الأسترالي", "GB2": "التشامبيونشيب", "QSL": "دوري نجوم قطر", "UAE1": "دوري أدنوك الإماراتي",
}

# FotMob league id -> (Transfermarkt-style code, calendar-year season?) used for the alternatives pool.
POOL_LEAGUES = {
    47: ("GB1", False), 87: ("ES1", False), 55: ("IT1", False), 54: ("L1", False), 53: ("FR1", False),
    536: ("SA1", False), 57: ("NL1", False), 61: ("PO1", False), 71: ("TR1", False), 40: ("BE1", False),
    64: ("SC1", False), 48: ("GB2", False), 535: ("QSL", False), 538: ("UAE1", False), 46: ("DK1", False),
    130: ("MLS1", True), 268: ("BRA1", True),
}
ARAB_LEAGUES = {"SA1", "QSL", "UAE1"}

# Transfermarkt sub_position -> position group
SUB_POSITION_GROUP = {
    "Goalkeeper": "GK",
    "Centre-Back": "CB",
    "Left-Back": "FB", "Right-Back": "FB",
    "Defensive Midfield": "DM",
    "Central Midfield": "CM",
    "Attacking Midfield": "AM",
    "Left Winger": "W", "Right Winger": "W", "Left Midfield": "W", "Right Midfield": "W",
    "Centre-Forward": "ST", "Second Striker": "ST",
}
POSITION_FALLBACK = {"Goalkeeper": "GK", "Defender": "CB", "Midfield": "CM", "Attack": "ST"}

POSITION_NAMES_AR = {
    "GK": "حارس مرمى", "CB": "قلب دفاع", "FB": "ظهير", "DM": "وسط دفاعي",
    "CM": "وسط", "AM": "صانع ألعاب", "W": "جناح", "ST": "مهاجم",
}
POSITION_SHORT = {"GK": "GK", "CB": "CB", "FB": "FB", "DM": "CDM", "CM": "CM", "AM": "CAM", "W": "W", "ST": "ST"}
OUTFIELD_GROUPS = ["ST", "W", "AM", "CM", "DM", "FB", "CB"]

# Attribute catalogue (FC-style). Order matters for display.
ATTRIBUTES = {
    "pace": "السرعة القصوى",
    "acceleration": "التسارع",
    "finishing": "الإنهاء",
    "composure": "الهدوء",
    "positioning": "التحرك بدون كرة",
    "passing": "التمرير",
    "vision": "الرؤية وصناعة اللعب",
    "dribbling": "المراوغة والتحكم",
    "defensive_awareness": "الوعي الدفاعي",
    "tackling": "الافتكاك والالتحامات",
    "tactical_iq": "الذكاء التكتيكي",
    "work_rate": "الضغط ومعدل العمل",
    "strength": "القوة البدنية",
    "stamina": "التحمل",
    "aerial": "الكرات الهوائية",
    "goalkeeping": "حراسة المرمى",
}

# How each position group's overall is weighted across attributes (each row sums to 1).
POSITION_WEIGHTS = {
    "ST": {"finishing": .22, "positioning": .16, "composure": .12, "pace": .10, "acceleration": .06,
           "dribbling": .10, "strength": .08, "aerial": .06, "work_rate": .05, "vision": .05},
    "W": {"pace": .14, "acceleration": .12, "dribbling": .18, "finishing": .12, "vision": .10,
          "passing": .08, "positioning": .08, "composure": .06, "work_rate": .06, "stamina": .06},
    "AM": {"vision": .20, "passing": .16, "dribbling": .14, "composure": .10, "finishing": .10,
           "positioning": .08, "tactical_iq": .10, "acceleration": .06, "work_rate": .06},
    "CM": {"passing": .18, "vision": .14, "tactical_iq": .14, "stamina": .10, "work_rate": .10,
           "composure": .08, "dribbling": .08, "defensive_awareness": .08, "tackling": .05, "strength": .05},
    "DM": {"defensive_awareness": .18, "tackling": .16, "tactical_iq": .16, "passing": .14, "strength": .08,
           "stamina": .08, "work_rate": .08, "composure": .07, "aerial": .05},
    "FB": {"pace": .14, "stamina": .12, "defensive_awareness": .14, "tackling": .12, "passing": .10,
           "acceleration": .08, "work_rate": .10, "dribbling": .08, "tactical_iq": .07, "vision": .05},
    "CB": {"defensive_awareness": .22, "tackling": .18, "aerial": .14, "strength": .14, "tactical_iq": .10,
           "composure": .07, "passing": .08, "pace": .07},
    "GK": {"goalkeeping": .70, "composure": .10, "passing": .10, "tactical_iq": .05, "aerial": .05},
}
