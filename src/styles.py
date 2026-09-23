"""IBM Carbon Design styling for CardioCore AI (dark "g100" theme, compact single-screen layout).

Based on the /carbon-streamlit skill template: a parameterised THEME dict plus a CSS generator.
Differences from the stock template: dark palette, and dashboard rules (tight padding, compact
widgets, metric tiles, tabs, classification badge, scrolling side panel) so the whole app fits one
screen without page scrolling.  Override colours with get_carbon_css({"primary": "#..."}).
"""

THEME: dict[str, str] = {
    "primary": "#4589ff",
    "primary_hover": "#0f62fe",
    "primary_active": "#002d9c",
    "bg_primary": "#161616",
    "bg_secondary": "#262626",
    "bg_tertiary": "#393939",
    "text_primary": "#f4f4f4",
    "text_secondary": "#c6c6c6",
    "text_helper": "#8d8d8d",
    "text_placeholder": "#6f6f6f",
    "border": "#393939",
    "success": "#42be65",
    "warning": "#f1c21b",
    "orange": "#ff832b",
    "error": "#fa4d56",
    "font_sans": "'IBM Plex Sans', sans-serif",
    "font_mono": "'IBM Plex Mono', monospace",
}

# Badge colour per rhythm class (Carbon status palette).
CLASS_COLORS = {
    "NSR": THEME["success"],
    "Sinus Bradycardia": THEME["primary"],
    "Sinus Tachycardia": THEME["warning"],
    "Atrial Fibrillation": THEME["error"],
    "PVC": THEME["orange"],
}


def get_carbon_css(theme: dict[str, str] | None = None) -> str:
    """Return a <style> block with the Carbon dark dashboard styles."""
    t = {**THEME, **(theme or {})}
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"], .stMarkdown, button, input, select, textarea {{
        font-family: {t["font_sans"]} !important;
    }}

    /* Hide Streamlit chrome and remove page scrolling: the dashboard must fit the viewport. */
    #MainMenu, header, footer {{ visibility: hidden; height: 0; }}
    .stApp {{ background-color: {t["bg_primary"]}; }}
    .stMainBlockContainer {{
        padding: 0.6rem 1.2rem 0.2rem 1.2rem !important;
        max-width: 100% !important;
    }}
    [data-testid="stVerticalBlock"] {{ gap: 0.45rem !important; }}
    [data-testid="stHorizontalBlock"] {{ gap: 0.75rem !important; }}

    /* Title */
    h1 {{
        font-size: 1.25rem !important; font-weight: 600 !important; letter-spacing: -0.01em;
        color: {t["text_primary"]} !important; padding: 0 !important; margin: 0 !important; line-height: 1.3 !important;
    }}
    .subtitle {{ font-size: 0.75rem; color: {t["text_helper"]}; margin-top: 0.3rem; line-height: 1.2; }}

    /* Compact widgets */
    [data-testid="stWidgetLabel"] p {{
        font-size: 0.7rem !important; color: {t["text_secondary"]} !important;
        text-transform: uppercase; letter-spacing: 0.04em;
    }}
    [data-testid="stWidgetLabel"] {{ min-height: 0 !important; }}
    [data-testid="stSlider"] {{ padding-bottom: 0 !important; }}
    [data-testid="stSlider"] > div {{ padding-top: 0 !important; }}
    div[data-baseweb="select"] > div {{
        background-color: {t["bg_secondary"]} !important; border-radius: 0 !important;
        border: none !important; border-bottom: 1px solid {t["text_helper"]} !important;
        min-height: 2rem !important; font-size: 0.85rem;
    }}

    /* Buttons: Carbon rectangular */
    .stButton > button, [data-testid="stPopover"] > button {{
        border-radius: 0 !important; font-size: 0.8rem !important; font-weight: 500 !important;
        min-height: 2rem !important; padding: 0.2rem 0.9rem !important; text-transform: none !important;
    }}
    .stButton > button[kind="primary"] {{
        background-color: {t["primary_hover"]} !important; color: #fff !important; border: none !important;
    }}
    .stButton > button[kind="primary"]:hover {{ background-color: {t["primary"]} !important; }}
    .stButton > button[kind="secondary"], [data-testid="stPopover"] > button {{
        background: transparent !important; color: {t["primary"]} !important;
        border: 1px solid {t["primary"]} !important;
    }}

    /* Segmented control (mode / view switches) */
    [data-testid="stSegmentedControl"] button {{ border-radius: 0 !important; font-size: 0.8rem !important; }}

    /* Metric tiles */
    [data-testid="stMetric"] {{
        background: {t["bg_secondary"]}; border-left: 3px solid {t["primary"]};
        padding: 0.35rem 0.7rem; min-height: 0;
    }}
    [data-testid="stMetricLabel"] p {{
        font-size: 0.66rem !important; color: {t["text_helper"]} !important;
        text-transform: uppercase; letter-spacing: 0.05em;
    }}
    [data-testid="stMetricValue"] {{ font-size: 1.15rem !important; font-weight: 500; }}
    [data-testid="stMetricValue"] div {{ line-height: 1.2; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid {t["border"]}; }}
    .stTabs [data-baseweb="tab"] {{
        height: 2.2rem; padding: 0 1rem; font-size: 0.85rem; color: {t["text_secondary"]};
    }}
    .stTabs [aria-selected="true"] {{ color: {t["text_primary"]}; }}

    /* Right-hand agent panel: fixed height, scrolls internally so the PAGE never scrolls. */
    .st-key-agent_panel {{
        height: calc(100vh - 8.9rem) !important; max-height: calc(100vh - 8.9rem) !important;
        overflow-y: auto !important; padding-right: 0.4rem; margin-top: 0.5rem;
    }}
    .st-key-agent_panel::-webkit-scrollbar {{ width: 6px; }}
    .st-key-agent_panel::-webkit-scrollbar-thumb {{ background: {t["bg_tertiary"]}; }}

    /* Classification badge */
    .badge {{
        border-left: 4px solid var(--c); background: {t["bg_secondary"]};
        padding: 0.45rem 0.8rem; margin-bottom: 0.2rem;
    }}
    .badge .b-title {{ font-size: 1.05rem; font-weight: 600; letter-spacing: 0.02em; color: var(--c); }}
    .badge .b-sub {{ font-size: 0.7rem; color: {t["text_helper"]}; margin-top: 0.1rem; }}

    /* Lesson cards */
    .card {{
        background: {t["bg_secondary"]}; padding: 0.55rem 0.8rem; margin-bottom: 0.4rem;
        border-left: 2px solid {t["bg_tertiary"]};
    }}
    .card .c-title {{
        font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.08em;
        color: {t["text_helper"]}; margin-bottom: 0.2rem;
    }}
    .card .c-body {{ font-size: 0.82rem; color: {t["text_primary"]}; line-height: 1.45; }}
    .card.flag {{ border-left-color: {t["error"]}; }}
    .card.quiz {{ border-left-color: {t["primary"]}; }}
    .wave {{ font-size: 0.8rem; color: {t["text_secondary"]}; margin: 0.15rem 0; line-height: 1.4; }}
    .wave b {{ color: {t["primary"]}; font-family: {t["font_mono"]}; font-weight: 500; }}

    /* Footer disclaimer pinned to the bottom of the viewport */
    .disclaimer {{
        position: fixed; bottom: 4px; left: 1.2rem; font-size: 0.66rem; color: {t["text_placeholder"]};
        z-index: 9;
    }}

    hr {{ border: none; border-top: 1px solid {t["border"]}; margin: 0.6rem 0; }}
    </style>
    """
