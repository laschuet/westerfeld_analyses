# General
INPUT_FILE = "../df_Fungi_19_F_Co_.csv"  # The path to the input of the network
LOOKUP_FILE = "../df_fungi.csv"  # The path to the input of the network
NODE_NAME = "Species"  # This can be Species, but also OTUs or ASVs / identifier

# Prepocessing
USE_ABSOLUTE_THRESHOLD_TO_OBTAIN_COMMON_ASVS = True  # Set to `True` if all ASVs, with a total sum of absoute abundances, that are less than a threshold, should be removed
ABSOLUTE_THRESHOLD = 100

# Caution: if activated, this will remove a lot of potential specialists
USE_ZERO_RATIO_TO_OBTAIN_COMMON_ASVS = (
    True  # Set to `True` if all avs with more than the given ration should be removed
)

ZERO_RATIO_THRESHOLD = 0.5  # Above -> will be removed
CONVERT_FROM_ABSOLUTE_TO_RELATIVE = True  # Set to `True` if the input data consists of absolute abundances and you want to convert to relatice abundances

USE_MCLR = (
    True  # Set to `TRUE` if the modified centered log ratio (mCLR) should be applied
)
MCLR_C = 1  # Minimal distance from zeros after mCLR was perfomed

# Specialists and generalists
MEAN_RELATIVE_ABUNDANCES_LOWER_THRESHOLD: float = 2e-5
B_VALUE_SPECIALIST_THRESHOLD = 1.5  # Values lower than `B_VALUE_SPECIALIST_THRESHOLD` are identified as specialists
B_VALUE_GENERALIST_THRESHOLD = 6.0  # Values larger than `B_VALUE_GENERALIST_THRESHOLD` are identified as generalists
