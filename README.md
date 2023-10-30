# Interpretable Machine Learning with Python

<a href="https://www.packtpub.com/product/interpretable-machine-learning-with-python-2e-second-edition/9781803235424"><img src="https://content.packt.com/B18406/cover_image_small.jpg" alt="Interpretable Machine Learning with Pythone" height="256px" align="right"></a>

This is the code repository for [Interpretable Machine Learning with Python, 2E](https://www.packtpub.com/product/interpretable-machine-learning-with-python-2e-second-edition/9781803235424), published by Packt.

**Build explainable, fair, and robust high-performance models with hands-on, real-world examples**

The author of this book is -[Serg Masís](https://www.linkedin.com/in/smasis/)

## About the book(long descriptio in EPIC)

Interpretable Machine Learning with Python, Second Edition, brings to light the key concepts of interpreting machine learning models by analyzing real-world data, providing you with a wide range of skills and tools to decipher the results of even the most complex models.
Build your interpretability toolkit with several use cases, from flight delay prediction to waste classification to COMPAS risk assessment scores. This book is full of useful techniques, introducing them to the right use case. Learn traditional methods, such as feature importance and partial dependence plots to integrated gradients for NLP interpretations and gradient-based attribution methods, such as saliency maps.
In addition to the step-by-step code, you’ll get hands-on with reducing complexity, mitigating bias, placing guardrails, and enhancing reliability.
By the end of the book, you’ll be confident in tackling interpretability challenges with black-box models using tabular, language, image, and time series data.

## What's New:
This second edition introduces analysis of NLP transformers, leveraging BertViz to visualize transformer models, layers, and attention heads, and integrated gradients and the Visualization Data Record to see what tokens are responsible for a predicted label. You’ll also get a crash course on the Learning Interpretability Tool (LIT) and go through new and updated uses case.


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


## Outline and Chapter Summary
The title of this book suggests its central themes: interpretation, machine learning, and Python, with the first theme being the most crucial.
So, why is interpretation so important?
Interpretable machine learning encompasses a growing array of techniques that help us glean insights from models, aiming to ensure they are safe, fair, and reliable – a goal I believe we all share for our models.
With the rise of AI superseding traditional software and even human tasks, machine learning models are viewed as a more advanced form of software. While they operate on binary data, they aren’t typical software; their logic isn’t explicitly coded by developers but emerges from data patterns. This is where interpretation steps in, helping us understand these models, pinpoint their errors, and rectify them before any potential mishaps. Thus, interpretation is essential in fostering trust and ethical considerations in these models. And it’s worth noting that in the not-so-distant future, training models might move away from coding to more intuitive drag-and-drop interfaces. In this context, understanding machine learning models becomes an invaluable skill.
Currently, there’s still a significant amount of coding involved in data preprocessing, exploration, model training, and deployment. And while this book is rich with Python examples, it’s not merely a coding guide removed from practical applications or the bigger picture. The book’s essence is to prioritize the why before the how when it comes to interpretable machine learning, as interpretation revolves around the question of why.
Therefore, most chapters of this book kickoff by outlining a mission (the why) and then delving into the methodology (the how). The aim is to achieve the mission using the techniques discussed in the chapter, with an emphasis on understanding the results. The chapters wrap up by pondering on the practical insights gained from the exercises.
The structure of this book is progressive, starting from the basics and moving to more intricate topics. The tools utilized in this book are open source and are products of leading research institutions like Microsoft, Google, and IBM. Even though interpretability is a vast research field with many aspects still in the developmental phase, this book doesn’t aim to cover it all. Its primary goal is to delve deeply into a selection of interpretability tools, making it beneficial for those working in the machine learning domain.
The book’s initial section introduces interpretability, emphasizing its significance in the business landscape and discussing its core components and challenges. The subsequent section provides a detailed overview of various interpretation techniques and their applications, whether it’s for classification, regression, tabular data, time series, images, or text. In the final section, readers will engage in practical exercises on model tuning and data training for interpretability, focusing on simplifying models, addressing biases, setting constraints, and ensuring dependability.
By the book’s conclusion, readers will be adept at using interpretability techniques to gain deeper insights into machine learning models.


1. Chapter 1, [Interpretation, Interpretability, and Explainability; and Why Does It All Matter?](https://github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python-2E/tree/main/01)
2. Chapter 2, [Key Concepts of Interpretability](https://github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python-2E/tree/main/02)
3. Chapter 3, [Interpretation Challenges](https://github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python-2E/tree/main/03)
4. Chapter 4, [Global Model-Agnostic Interpretation Methods](https://github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python-2E/tree/main/04)
5. Chapter 5, [Local Model-Agnostic Interpretation Methods](https://github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python-2E/tree/main/05)
6. Chapter 6, [Anchors and Counterfactual Explanations](https://github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python-2E/tree/main/06)
7. Chapter 7, [Visualizing Convolutional Neural Networks](https://github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python-2E/tree/main/07)
8. Chapter 8, [Interpreting NLP Transformers](https://github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python-2E/tree/main/08)
9. Chapter 9, [Interpretation Methods for Multivariate Forecasting and Sensitivity Analysis](https://github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python-2E/tree/main/09)
10. Chapter 10, [Feature Selection and Engineering for Interpretability](https://github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python-2E/tree/main/10)
11. Chapter 11, [Bias Mitigation and Causal Inference Methods](https://github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python-2E/tree/main/11)
12. Chapter 12, [Monotonic Constraints and Model Tuning for Interpretability](https://github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python-2E/tree/main/12)
13. Chapter 13, [Adversarial Robustness](https://github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python-2E/tree/main/13)
14. Chapter 14, What’s Next for Machine Learning Interpretability?

### Chapter 01, Interpretation, Interpretability, and Explainability; and Why Does It All Matter?
We live in a world whose rules and procedures are ever-increasingly governed by data and algorithms.
For instance, there are rules about who gets approved for credit or released on bail, and which social media posts might get censored. There are also procedures to determine which marketing tactics are most effective and which chest x-ray features might diagnose a positive case of pneumonia.
We expect this because it is nothing new!
But not so long ago, rules and procedures such as these used to be hardcoded into software, textbooks, and paper forms, and humans were the ultimate decision-makers. Often, it was entirely up to human discretion. Decisions depended on human discretion because rules and procedures were rigid and, therefore, not always applicable. There were always exceptions, so a human was needed to make them.
For example, if you apply for a mortgage, your approval depended on an acceptable and reasonably lengthy credit history. This data, in turn, would produce a credit score using a scoring algorithm. Then, the bank had rules that determined what score was good enough for the mortgage you wanted. Your loan officer could follow it or not.
These days, financial institutions train models on thousands of mortgage outcomes, with dozens of variables. These models can be used to determine the likelihood that you would default on a mortgage with a presumed high accuracy. If there is a loan officer to stamp the approval or denial, it’s no longer merely a guideline but an algorithmic decision. How could it be wrong? How could it be right? How and why was the decision made?
Hold on to that thought because, throughout this book, we will be learning the answers to these questions and many more!
Machine learning model interpretation enables you to understand the logic behind a decision and trace back the detailed steps of the process behind the logic. This chapter introduces machine learning interpretation and related concepts, such as interpretability, explainability, black-box models, and transparency. This chapter provides definitions for these terms to avoid ambiguity and underpins the value of machine learning interpretability. These are the main topics we are going to cover:
* What is machine learning interpretation?
* Understanding the difference between interpretation and explainability
* A business case for interpretability


### Chapter 02, Key Concepts of Interpretability
This book covers many model interpretation methods. Some produce metrics, others create visuals, and some do both; some depict models broadly and others granularly. In this chapter, we will learn about two methods, feature importance and decision regions, as well as the taxonomies used to describe these methods. We will also detail what elements hinder machine learning interpretability as a primer to what lies ahead.
The following are the main topics we are going to cover in this chapter:
* Learning about interpretation method types and scopes
* Appreciating what hinders machine learning interpretability

### Chapter 03, Interpretation Challenges


**Key Insights**:

### Chapter 04, Global Model-Agnostic Interpretation Methods


**Key Insights**:

### Chapter 05, Local Model-Agnostic Interpretation Methods


**Key Insights**:

### Chapter 06, Anchors and Counterfactual Explanations


**Key Insights**:

### Chapter 07, Visualizing Convolutional Neural Networks


**Key Insights**:


### Chapter 08, Interpreting NLP Transformers


**Key Insights**:

### Chapter 09, Interpretation Methods for Multivariate Forecasting and Sensitivity Analysis


**Key Insights**:

### Chapter 10, Feature Selection and Engineering for Interpretability


**Key Insights**:

### Chapter 11, Bias Mitigation and Causal Inference Methods


**Key Insights**:

### Chapter 12, Monotonic Constraints and Model Tuning for Interpretability


**Key Insights**:

### Chapter 13, Adversarial Robustness


**Key Insights**:

### Chapter 14, What’s Next for Machine Learning Interpretability?


**Key Insights**:



## Know more on the Discord server <img alt="Coding" height="25" width="32"  src="https://cliply.co/wp-content/uploads/2021/08/372108630_DISCORD_LOGO_400.gif">
You can get more engaged on the discord server for more latest updates and discussions in the community at [Discord](https://packt.link/inml)

## Download a free PDF <img alt="Coding" height="25" width="40" src="https://emergency.com.au/wp-content/uploads/2021/03/free.gif">

_If you have already purchased a print or Kindle version of this book, you can get a DRM-free PDF version at no cost. Simply click on the link to claim your free PDF._
[Free-Ebook](https://packt.link/free-ebook/9781803235424) <img alt="Coding" height="15" width="35"  src="https://media.tenor.com/ex_HDD_k5P8AAAAi/habbo-habbohotel.gif">

We also provide a PDF file that has color images of the screenshots/diagrams used in this book at [GraphicBundle](https://packt.link/gbp/9781803235424) <img alt="Coding" height="15" width="35"  src="https://media.tenor.com/ex_HDD_k5P8AAAAi/habbo-habbohotel.gif">



## Get to Know the Author
For the last two decades, _Serg Masís_ has been at the confluence of the internet, application development, and analytics. Currently, he's a Lead Data Scientist at Syngenta, a leading agribusiness company with a mission to improve global food security. Before that role, he co-founded a search engine startup incubated by Harvard Innovation Labs, combining the power of cloud computing and machine learning with principles in decision-making science to expose users to new places and events efficiently. Whether it pertains to leisure activities, plant diseases, or customer lifetime value, Serg is passionate about providing the often-missing link between data and decision-making — and machine learning interpretation helps bridge this gap more robustly



## Other Related Books
- [Causal Inference and Discovery in Python](https://www.packtpub.com/product/causal-inference-and-discovery-in-python/9781804612989)
- [Transformers for Natural Language Processing – Second Edition](https://www.packtpub.com/product/transformers-for-natural-language-processing-second-edition-second-edition/9781803247335)

