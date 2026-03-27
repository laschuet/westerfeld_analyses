# General
INPUT_FILE = "data/df_2019_fs.csv"  # The path to the input of the network
LOOKUP_FILE = "data/df_fungi.csv"  # The path to the input of the network
NODE_NAME = "Species"  # This can be Species, but also OTUs or AVSs / identifier

# Graph creation method
GRAPH_CREATOR_NAME = "correlation"  # Either: "correlation", "inference_glasso"
CORRELATION_COEFFICIENT = "spearman"  # Choose either 'spearman' or 'pearson'.

INVERSE_VARIANCE_ZERO_THRESHOLD = 1e-2  # If the inverse variance is smaller than this threshold, it will be set to zero.
GLASSO_ALPHAS = 7  # The alpha parameter for the graphical lasso. Higher values lead to more sparse networks.
GLASSO_MAX_ITER = 500  # The maximum number of iterations for the graphical lasso.

# Prepocessing
USE_ABSOLUTE_THRESHOLD_TO_OBTAIN_COMMON_ASVS = True  # Set to `TRUE` if all avs, with a total sum of absoute abundances, that are less than a threshold, should be removed
ABSOLUTE_THRESHOLD = 100

# Caution: if activated, this will remove a lot of potential sppecialists
USE_ZERO_RATION_TO_OBTAIN_COMMON_ASVS = (
    True  # Set to `TRUE` if all avs with more than the given ration should be removed
)

ZERO_RATION_THRESHOLD = 0.5  # Above -> will be removed
CONVERT_FROM_ABSOLUTE_TO_RELATIVE = True  # Set to `TRUE` if the input data consists of absolute abundances and you want to convert to relatice abundances.

USE_MCLR = True  # Set to `TRUE` if the modified centered log ration should be applied
MCLR_C = 1  # Minimal distance from zeros after mclr was perfomed

# Threshold
CORRELATION_THRESHOLD = (
    0.68  # Will be used a an absolute threshold (positive and negative)
)

# Specialists and generalists
MEAN_RELATIVE_ABUNDANCES_LOWER_THRESHOLD: float = 2e-5
B_VALUE_SPECIALIST_THRESHOLD: float = (
    1.5  # Lower than `B_VALUE_SPECIALIST_THRESHOLD` are identified as specialists
)
B_VALUE_GENERALIST_THRESHOLD: float = (
    6.0  # More than `B_VALUE_GENERALIST_THRESHOLD` are identified as generalists
)

# Kernels
USE_PHYLUM_LABELS_FOR_WEISFEILER_LEHMAN_KERNEL = True  # Set to `TRUE` if you want to use the phylum labels for the kernels, otherwise, the species names will be used as labels.

