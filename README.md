# PerturbationAUC

## TODO:

- [ ] Factuality critic (make it an optional strategy?)
- [ ] Strategy for handling separation of factual and counterfactual instructions & factual/counterfactual scores (`perturbation_auc_result["factual"]`?)
- [ ] Writing prompts and vibe-checking the system
- [ ] Optional pluggable(?) offspring-selection policy for when generated mutations overshoots `n_offspring_per_generation` (e.g., Levenshtein-diversity)