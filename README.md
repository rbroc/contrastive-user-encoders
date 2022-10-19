## Training transformers with structured context
Includes code for constrastive user encoder from the EMNLP Findings paper:
- Rocca, R., & Yarkoni, T. (2022), Language models as user encoders: Self-supervised learning of user encodings using transformers, to appear in *Findings of the Association for Computational Linguistics: EMNLP 2022* (link coming soon)

### Structure
- This repository does not include data, but the dataset can be recreated entirely using scripts made available under `reddit/preprocessing`;
- Model classes, trainer, and other utils can be found under `reddit`;
- `notebooks` include the code needed to replicate plots presented in the paper, as well as baseline fitting;
- `scripts` contain Python training scripts for both triplet loss training and downstream tasks;

Note: triplet loss training could be streamlined using HuggingFace's `transformers` library - future refactoring may simplify the current code in this direction.