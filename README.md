# PerturbationAUC

## TODO:

- [ ] Factuality critic (make it an optional strategy?)
- [ ] Don't set up/tear down an asyncio.gather event loop for every generation (parallelization of `PerturbationAUC.run`s in general?)
- [ ] Strategy for handling separation of factual and counterfactual instructions & factual/counterfactual scores (`perturbation_auc_result["factual"]`?)
- [ ] Writing prompts and vibe-checking the system