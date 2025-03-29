# metaphlan_tools/__init__.py

from .parser import (
    parse_metaphlan_file,
    combine_samples,
    load_metadata,
    join_abundance_with_metadata
)

from .stats import (
    calculate_alpha_diversity,
    compare_alpha_diversity,
    calculate_beta_diversity,
    perform_permanova,
    differential_abundance_analysis,
    plot_ordination
)

from .viz import (
    plot_relative_abundance_heatmap,
    plot_alpha_diversity_boxplot,
    plot_stacked_bar,
    plot_longitudinal_changes,
    plot_correlation_network,
    create_abundance_summary
)

__version__ = "0.1.0"
