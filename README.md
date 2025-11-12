### Linear Features Represent the Instrumentality of Goal Pursuit in Transformers

#### Key Findings



#### Dataset Generation
##### Contrastive Prompt Dataset
The Jupyter Notebook (dataset_generation.ipynb) was used to generate a contrastive dataset, through batched prompting of a capable open-source model (tngtech/deepseek-r1t2-chimera:free) using the OpenRouter API. In order to combat "creativity collapse" in the prompt dataset generation model, we employ a seeding method, where each batched call to the generation model is seeded with a unique (instrumental drive, task, scenario) tuple sampled from a hard-coded cross product set.

An Example from the Contrastive Dataset: 
```json

```

###### Discussion of parallel prompt datasets (Base vs Instruct)



##### Probe Training Dataset (Activation Dataset)

#### TODO
-[ ] Rerun prompt dataset curation for Base model (document style prompts)
-[ ] Gather activation data for both models w.r.t. native prompting style for every layer in the network to compose a training dataset for the probe
-[ ] Train the probes
-[ ] Cross model/prompting evaluation, etc.
-[ ] Steering effects!