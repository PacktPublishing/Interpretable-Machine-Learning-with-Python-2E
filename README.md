# Interpretable Machine Learning with Python

<a href="https://www.packtpub.com/product/interpretable-machine-learning-with-python-2e-second-edition/9781803235424"><img src="https://content.packt.com/B18406/cover_image_small.jpg" alt="Interpretable Machine Learning with Pythone" height="256px" align="right"></a>

This is the code repository for [Interpretable Machine Learning with Python, 2E](https://www.packtpub.com/product/interpretable-machine-learning-with-python-2e-second-edition/9781803235424), published by Packt.

**Build explainable, fair, and robust high-performance models with hands-on, real-world examples**

## What is this book about?
Do you want to gain a deeper understanding of your models and better mitigate poor prediction risks associated with machine learning interpretation? If so, then Interpretable Machine Learning with Python, Second Edition is the book for you.

This book covers the following exciting features:
* Recognize the importance of interpretability in business
* Study models that are intrinsically interpretable such as linear models, decision trees, Naive Bayes, and glass-box models, such as EBM and Gami-NET
* Become well-versed in interpreting black-box models with model-agnostic methods
* Use monotonic and interaction constraints to make fairer and safer models
* Understand how to mitigate the influence of bias in datasets
* Discover how to make models more reliable with adversarial robustness
* Understand how transformer models work and how to interpret them

If you feel this book is for you, get your [copy](https://www.amazon.com/Interpretable-Machine-Learning-Python-hands/dp/180323542X) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png"
alt="https://www.packtpub.com/" border="5" /></a>

## Instructions and Navigations
All of the code is organized into folders. For example, [02](/02) for Chapter 2.

The code will look like the following:
```
base_classifier = KerasClassifier(model=base_model,\
                                  clip_values=(min_, max_))
y_test_mdsample_prob = np.max(y_test_prob[sampl_md_idxs],\
                                                       axis=1)
y_test_smsample_prob = np.max(y_test_prob[sampl_sm_idxs],\
                                                       axis=1)
```

**Following is what you need for this book:**
This book is for data scientists, machine learning developers, MLOps engineers, and data stewards who have an increasingly critical responsibility to explain how the AI systems they develop work, their impact on decision making, and how they identify and manage bias It’s also a useful resource for self-taught ML enthusiasts and beginners who want to go deeper into the subject matter, though a good grasp of the Python programming language is needed to implement the examples.

With the following software and hardware list you can run all code files present in the book (Chapter 1-14).

### Software and Hardware List

#### Software

You can install the software required in any operating system by first installing [Jupyter Notebook or Jupyter Lab](https://jupyter.readthedocs.io/en/latest/install.html) with the most recent version of Python, or install [Anaconda](https://docs.anaconda.com/anaconda/) which can install everything at once. Python versions 3.9 and onwards are supported. For what packages are required see below.

#### Hardware

While hardware requirements for Jupyter are relatively modest, we recommend a machine with at least 4 cores of 2Ghz and 8Gb of RAM. A GPU is highly recommended for chapters with Deep Learning ([7](/07), [8](/08), [9](/09), [12](/12), and [13](/13)). If you don't have a GPU in your machine, we recommend you use [Google Colab](https://colab.research.google.com/) for these chapters. Skip to the Google Colab section for details.

#### Python Packages

The packages required for the entire book are in [requirements.txt](requirements.txt) and there's a similar file in in each chapter folder specifying the requirements for the corresponding chapter. For installation instructions, see below.

### Installation Instructions

Once you clone this repository, there are many ways to setup your Python environment and Jupyter installation. You may have your packages managed by Anaconda or Miniconda (`conda`), Conan (`conan`), Mamba (`mamba`), or Poetry (`poetry`). Either way, we strongly suggest you setup an independent environment and install the requirements there with `python` like this:

``` sh
# Go to cloned repository
cd /path/to/repo/

# Run the setup script
python setup.py
```

Under the hood it is using `pip` to install the packages one-by-one as specified [requirements.txt](requirements.txt).
If it's starting from a clean slate it should have no conflicts. Please note that for some systems you might need to install `pip` and `setuptools` before you run `setup.py` since these usually come already installed but sometimes they aren't.

If you are having issues with the installation you can try pip directly which can install most of the packages all-in-one-go using `-r`. There are two packages we must install without dependencies to avoid conflicts.

``` sh
# Install most requirements (it will fail if one conflict is found)
pip install -r requirements_w_pip.txt

# Install the two requirements with special arguments
pip install --no-deps git+https://github.com/EthicalML/xai.git
pip install --no-deps adversarial-robustness-toolbox~=1.12.2
```

You can also create virtual environments on a chapter by chaper basis with `pipenv`, `venv`, `virtualenv`, `conda` or any of the other options, and use the `chapter_setup.py` script located in every chapter to ensure that the packages are installed. For instance:

``` sh
# Go to chapter's folder
cd /path/to/chapter

# Create a virtual environment called "myenv"
python -m venv myvenv

# Activate the "myenv" environment
source myenv/bin/activate

# Run the chapter setup script in that environment
python chapter_setup.py
```

The setup script won't take care of your Jupyter installation, You would still need to install Jupyter like this:

``` sh
# Install Jupyter Notebook
pip install notebook

# Set the "myenv" virtual environment as a Jupyter kernel
python -m ipykernel install --user --name=myvenv

# Start a Jupyter server (typically available at http://localhost:8888/tree)
jupyter notebook
```

### Google Colab

You can run  all the code in Google Colab with the following links:

* **Chapter 1**: _Interpretation, Interpretability, and Explainability; and why does it all matter?_
  * Weight Prediction [WeightPrediction.ipynb](https://drive.google.com/file/d/17XtIBNrvvbQSKA9jAIbSzcwaRQn8XtmK/view?usp=sharing)
* **Chapter 2**: _Key Concepts of Interpretability_
  * Cardiovascular Disease Risk Factors [CVD.ipynb](https://drive.google.com/file/d/12W2NkwK0YEWhu0ZCE-WJ7ruJQbLpcAHP/view?usp=sharing)
* **Chapter 3**: _Interpretation Challenges_
  * Flight Delay Prediction [FlightDelays.ipynb](https://drive.google.com/file/d/1UBdR2HhyvXXV0vTIMASSPBEkiuSk32E7/view?usp=sharing) (Best run on CPU with High RAM)
* **Chapter 4**: _Global Model-Agnostic Interpretation Methods_
  * Used Cars Price Prediction [UsedCars.ipynb](https://colab.research.google.com/drive/1G_haMEW9NM3NBrY9WMlNY6D9Hi_g_Ef0?usp=sharing) (Best run on CPU with High RAM)
* **Chapter 5**: _Local Model-agnostic Interpretation Methods_
  * Chocolate Rating Classification [ChocoRatings.ipynb](https://colab.research.google.com/drive/1vvvyYiuzYp0uGQIhfMqf-0V15gKfnSoZ?usp=sharing)
* **Chapter 6**: _Anchor and Counterfactual Explanations_
  * Criminal Recidivism Prediction [Recidivism.ipynb](https://colab.research.google.com/drive/1g7EDH8FIOUurTjFmO6vYAUvs3_Us8pmZ?usp=sharing)
* **Chapter 7**: _Visualizing Convolutional Neural Networks_
  * Garbage Classification [GarbageClassifier.ipynb](https://colab.research.google.com/drive/1dEP2rdTqsPjZ2ANT-wqkZ2lNAjTiTy6T?usp=sharing) (Best run on CPU with High RAM)
* **Chapter 8**: _Interpreting NLP Transformers_
  * Restaurant Review Sentiment Analysis [ReviewSentiment.ipynb](https://colab.research.google.com/drive/1_lBFkQba5fue09WR1a7QA5-CmKYq1wl3?usp=sharing)
* **Chapter 9**: _Interpretation Methods for Multivariate Forecasting and Sensitivity Analysis_
  * Traffic Forecasting: Main Example [Traffic_compact1.ipynb](https://colab.research.google.com/drive/1WOrNkAqglWRzQIaeqo1Cnu1HHYnEiJsx?usp=sharing)
  * Traffic Forecasting: Alternative Example [Traffic_compact2.ipynb](https://colab.research.google.com/drive/1sDCSNartnn9HCRoGXIx6YtWzYEz3MHBb?usp=sharing)
  * Traffic Forecasting: Alternative Example [Traffic_optimal.ipynb](https://colab.research.google.com/drive/1XOAfgIpKOT_WWK7MMxZUkVcVZsq5Jws1?usp=sharing)
* **Chapter 10**: _Feature Selection and Engineering for Interpretability_
  * Non-Profit Donation Prediction [Mailer.ipynb](https://colab.research.google.com/drive/1aoVToakCwNrACvjp_B0XwViE6fbv8Okz?usp=sharing) (Best run on CPU with High RAM)
* **Chapter 11**: _Bias Mitigation and Causal Inference Methods_
  * Loan Default Classification [CreditCardDefaults.ipynb](https://drive.google.com/file/d/1ZZmGiUZsnik3o9PDJYUbeNIzksV4kMqD/view?usp=sharing)
* **Chapter 12**: _Monotonic Constraints and Model Tuning for Interpretability_
  * Criminal Recidivism Prediction (part 2) [Recidivism_part2.ipynb](https://drive.google.com/file/d/1VMeG10cqQhgoWbQ3lwEiINPV-ZtezP2h/view?usp=sharing) (Best run on CPU with High RAM)
* **Chapter 13**: _Adversarial Robustness_
  * Face Mask Classification [Masks.ipynb](https://drive.google.com/file/d/1VsBxTCK-8z6qqaZsohkMGt2i37wfqvbx/view?usp=sharing)

Remember to make sure you click on the menu item __"File > Save a copy in Drive"__ as soon you open each link to ensure that your notebook is saved as you run it. Also, some chapters are relatively memory-intensive, and will take  an extremely long time to run on Google Colab unless you go to __"Runtime > Change runtime type"__ and select __"High-RAM"__ for runtime shape. Otherwise, a better cloud enviornment or local environment is preferable.

## Troubleshooting

- At the time of this writing, the `lightgbm` package is difficult to compile for Mac computers with Apple Sillicon chips (M1/M2). Please refer to their [documentation](https://github.com/microsoft/LightGBM) for detailed instructions. However, `gaminet` package binaries are incompatible with Apple Sillicon so the final section of Chapter 3 won't work on Mac computers with that architecture.
- The widget for `lit-nlp` in Chapter 8 won't work in Colab environment and sometimes permissions issues get in the way to it working locally. Please refer to their [documentation](https://github.com/PAIR-code/lit) for detailed instructions on how to address that.
- The `aif360` package "Equalized Odds Postprocessing" method has an incompatibility with the `numpy` library that has been addressed in Chapter 11 with a "Monkey Patch". At the time of this writing, it has been fixed in the [AIF360 repository](https://github.com/Trusted-AI/AIF360/pull/458/commits/b5f2c51406acfc35508eb3a7d7ce9cd8b1a1d485) but not yet released. When it does, the patch will no longer be needed.
- If you have an issue with a package incompatibility in one or more of the notebooks, first make sure you have the right packages installed by inspecting all the installed packages with:

``` sh
pip list
```

- If you have an issue to report with one or more of the notebooks, please get the list of packages you have installed first to attach to the issue like this:

``` sh
pip freeze > installed.txt
```

- If you have an issue with the `machine-learning-datasets` library (also known as `mldatasets`), please report the issue in the [corresponding repository](https://github.com/smasis001/Machine-Learning-Datasets).


## Get to Know the Authors
**Serg Masís**
For the last two decades, Serg Masís has been at the confluence of the internet, application development, and analytics. Currently, he's a Lead Data Scientist at Syngenta, a leading agribusiness company with a mission to improve global food security. Before that role, he co-founded a search engine startup incubated by Harvard Innovation Labs, combining the power of cloud computing and machine learning with principles in decision-making science to expose users to new places and events efficiently. Whether it pertains to leisure activities, plant diseases, or customer lifetime value, Serg is passionate about providing the often-missing link between data and decision-making — and machine learning interpretation helps bridge this gap more robustly
