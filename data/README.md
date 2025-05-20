# Dataset Construction
## Original Datasets
To construct the original datasets,please refer to `generate_raw_{wikidata,celebrity}.py`.

We also provide intermediate outputs and the final files `wikidata_{train,test}_free.jsonl`, `celebrity_{train,test}_free.jsonl`.

## Continuation Datasets
We construct model-specific continuation datasets as follows:
* **Positive Examples:** Prompt the model to answer positive verification questions (see  `scripts/generate_{wikidata,celebrity}.sh`).
* **Negative Examples:**
    1. Use temperature sampling to get the model's self-generated answers.
  2. Prompt the model to answer negative verification questions (also in `scripts/generate_{wikidata,celebrity}.sh`).
* Run `generate_{wikidata,celebrity}.py` to generate the full continuation datasets.