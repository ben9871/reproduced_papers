import warnings

# Matplotlib relies on pyparsing APIs that emit deprecation warnings under pytest.
warnings.filterwarnings(
    "ignore",
    message=".*deprecated.*",
    module="matplotlib._fontconfig_pattern",
)
warnings.filterwarnings(
    "ignore",
    message=".*enablePackrat.*deprecated.*",
    module="matplotlib._mathtext",
)
