"""AuraPro Engine Pro model identifiers and the upstream quantizations they select."""

MODELS = {
    "IQ1_M": ("unsloth", "UD-IQ1_M"),
    "Q2_0": ("unsloth", "UD-Q2_0"),
    "IQ2_XS": ("unsloth", "UD-IQ2_XS"),
    "Q2_K_XL": ("unsloth", "UD-Q2_K_XL"),
    "IQ3_XXS": ("unsloth", "UD-IQ3_XXS"),
    "Q3_K_XL": ("unsloth", "UD-Q3_K_XL"),
    "IQ4_XS": ("unsloth", "UD-IQ4_XS-v31"),
    "lite_Q4": ("qwen36", "UD-Q4_K_XL"),
    "lite_Q6": ("qwen36", "UD-Q6_K_XL"),
}
DEFAULT_MODEL = "Q2_0"
