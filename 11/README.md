# CHAPTER 11
**_Bias Mitigation and Causal Inference Methods_**

## How to Run

Open "CreditCardDefaults.ipynb" notebook in a local Jupyter environment or open notebook in [Google Colab](https://drive.google.com/file/d/1ZZmGiUZsnik3o9PDJYUbeNIzksV4kMqD/view?usp=sharing)

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
lightgbm~=4.0.0
networkx~=3.2
pydot~=1.4.2
tqdm~=4.66.1
ipython>=7.23.0
machine-learning-datasets~=0.1.23
xgboost~=2.0.0
#xgboost>=1.7.6
econml~=0.14.1
dowhy~=0.10.1
aif360~=0.5.0
fairlearn~=0.7
BlackBoxAuditing~=0.1.54
git+https://github.com/EthicalML/xai.git # --no-deps
```