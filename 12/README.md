# CHAPTER 12
**_Monotonic Constraints and Model Tuning for Interpretability_**

## How to Run

Open "Recidivism_part2.ipynb" notebook in a local Jupyter environment or open notebook in [Google Colab](https://drive.google.com/file/d/1VMeG10cqQhgoWbQ3lwEiINPV-ZtezP2h/view?usp=sharing) (Best run on CPU with High RAM)

When you run it locally to install the requirements, you can run `chapter_setup.py` from the terminal like this:

``` sh
python chapter_setup.py
```

But you can also do it from the first cell of the notebook.

## Requirements

For your reference, these are the requirements:

```
pandas~=1.5.3
numpy~=1.23.5
scikit-learn~=1.2.2
matplotlib~=3.7.1
seaborn~=0.12.2
scipy~=1.11.3
tensorflow~=2.14.0
lightgbm~=4.0.0
machine-learning-datasets~=0.1.23
xgboost~=2.0.0
catboost~=1.2
shap~=0.42.1
ray~=2.7.1
alibi~=0.9.4
aif360~=0.5.0
bayesian-optimization~=1.4.3
scikeras[tensorflow]~=0.11.0
tensorflow-lattice~=2.0.13
git+https://github.com/EthicalML/xai.git # --no-deps
```