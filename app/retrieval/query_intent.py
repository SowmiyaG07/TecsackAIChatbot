from __future__ import annotations

import re


# ---------------------------------------------------------
# Intent keyword configuration
# ---------------------------------------------------------

# Keywords are matched as whole words/phrases (case-insensitive),
# so include plural forms explicitly.

HR_KEYWORDS = [
    "hr",
    "human resource",
    "human resources",
    "holiday",
    "holidays",
    "holiday list",
    "public holiday",
    "leave",
    "leaves",
    "leave policy",
    "sick leave",
    "casual leave",
    "earned leave",
    "maternity",
    "paternity",
    "vacation",
    "payslip",
    "payslips",
    "pay slip",
    "salary",
    "payroll",
    "attendance",
    "working hours",
    "work from home",
    "wfh",
    "notice period",
    "resignation",
    "probation",
    "appraisal",
    "reimbursement",
    "handbook",
    "employee handbook",
    "code of conduct",
    "policy",
    "policies",
    "hr policy",
    "employee policy",
    "resolution",
]

# One entry per project in data/Projects (plus common spellings).
# Update this list when a new project folder/document is added.
PROJECT_KEYWORDS = [
    # Enterprise
    "apogee",
    "indigo",
    "digitalsavings",
    "digital savings",
    "synergyapp",
    "synergy app",
    "synergy",
    "fieldster",
    "peopleapps",
    "people apps",
    "school feeding",
    "plexo",
    "sales rep portal",
    "pooling automation",
    "poolingautomation",
    # Healthcare
    "picom365",
    "picom",
    # Inspection
    "primeinspect",
    "prime inspect",
    # IoT
    "terra",
    "scentplus",
    "scent plus",
    "titan printer",
    # Loyalty programs
    "coffeecup",
    "coffee cup",
    "euromilan",
    "euro milan",
    "muno",
    "poln8",
    "tabletoorder",
    "table to order",
    "virtualterminal",
    "virtual terminal",
    "gift card",
    "giftcard",
    "loyalty program",
]

# Weaker signals: only used when no HR/company keyword matched.
PROJECT_GENERIC_KEYWORDS = [
    "project",
    "projects",
    "case study",
    "case studies",
    "client",
    "clients",
    "portfolio",
]

COMPANY_KEYWORDS = [
    "about tecsack",
    "about tecsack software",
    "what does tecsack do",
    "what does tecsack software do",
    "company overview",
    "company profile",
    "company mission",
    "company vision",
    "company services",
    "who is tecsack",
    "tell me about tecsack",
    "explain about tecsack",
    "ceo of tecsack",
    "founder of tecsack",
    "team members of tecsack",
    "team member of tecsack",
    "leadership team of tecsack",
]


# ---------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------

def _compile_keywords(keywords: list[str]) -> re.Pattern[str]:
    """
    Build one regex that matches any keyword as a whole
    word/phrase. Longer phrases are tried first.
    """

    alternatives = sorted(
        (re.escape(keyword.lower()) for keyword in keywords),
        key=len,
        reverse=True,
    )

    return re.compile(
        r"\b(?:" + "|".join(alternatives) + r")\b"
    )


HR_PATTERN = _compile_keywords(HR_KEYWORDS)
PROJECT_PATTERN = _compile_keywords(PROJECT_KEYWORDS)
PROJECT_GENERIC_PATTERN = _compile_keywords(PROJECT_GENERIC_KEYWORDS)
COMPANY_PATTERN = _compile_keywords(COMPANY_KEYWORDS)


# ---------------------------------------------------------
# Query intent classification
# ---------------------------------------------------------

def classify_query_intent(
    question: str,
) -> tuple[str, str]:
    """
    Classify a user question into a broad retrieval intent.

    Returns:
        tuple[str, str]:
            intent
            justification

    Intents:
        company
        hr
        project
        general
    """

    question_lower = question.lower().strip()

    if not question_lower:
        return (
            "general",
            "The question is empty, so no specific document "
            "category can be identified.",
        )

    # -----------------------------------------------------
    # Project (named project - strongest signal)
    # -----------------------------------------------------
    #
    # Checked first so "CoffeeCup refund policy" is not
    # routed to HR by the word "policy".

    if PROJECT_PATTERN.search(question_lower):
        return (
            "project",
            "The question refers to a specific Tecsack "
            "project or project-related product.",
        )

    # -----------------------------------------------------
    # HR
    # -----------------------------------------------------

    if HR_PATTERN.search(question_lower):
        return (
            "hr",
            "The question matches HR or employee-related "
            "information such as leave, holidays, salary, "
            "payslips, or HR policies.",
        )

    # -----------------------------------------------------
    # Company
    # -----------------------------------------------------

    if COMPANY_PATTERN.search(question_lower):
        return (
            "company",
            "The question asks about Tecsack Software at "
            "the company level, such as its overview, "
            "leadership, team, mission, or services.",
        )

    # -----------------------------------------------------
    # Project (generic wording)
    # -----------------------------------------------------

    if PROJECT_GENERIC_PATTERN.search(question_lower):
        return (
            "project",
            "The question asks about Tecsack projects, "
            "clients, or case studies in general.",
        )

    # -----------------------------------------------------
    # General
    # -----------------------------------------------------

    return (
        "general",
        "The question does not contain enough high-confidence "
        "information to select a specific document category.",
    )


# ---------------------------------------------------------
# Metadata filter
# ---------------------------------------------------------

def get_metadata_filter(
    intent: str,
) -> dict[str, str] | None:
    """
    Convert query intent into a ChromaDB metadata filter.
    """

    if intent == "company":
        return {"top_level": "company"}

    if intent == "hr":
        return {"top_level": "hr"}

    if intent == "project":
        return {"top_level": "project"}

    return None