# Interpretable Machine Learning with Python

<a href="https://www.packtpub.com/product/interpretable-machine-learning-with-python-2e-second-edition/9781803235424"><img src="https://content.packt.com/B18406/cover_image_small.jpg" alt="Interpretable Machine Learning with Pythone" height="256px" align="right"></a>

This is the code repository for [Interpretable Machine Learning with Python, 2E](https://www.packtpub.com/product/interpretable-machine-learning-with-python-2e-second-edition/9781803235424), published by Packt.

**Build explainable, fair, and robust high-performance models with hands-on, real-world examples**

The author of this book is -[Serg Masís](https://www.linkedin.com/in/smasis/)

## About the book

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
In this chapter, we will discuss the traditional methods used for machine learning interpretation for both regression and classification. This includes model performance evaluation methods such as RMSE, R-squared, AUC, ROC curves, and the many metrics derived from confusion matrices. We will then examine the limitations of these performance metrics and explain what exactly makes “white-box” models intrinsically interpretable and why we cannot always use white-box models. To answer these questions, we’ll consider the trade-off between prediction performance and model interpretability. Finally, we will discover some new “glass-box” models such as Explainable Boosting Machines (EBMs) and GAMI-Net that attempt to not compromise on this trade-off between predictive performance and interpretability.
The following are the main topics that will be covered in this chapter:
* Reviewing traditional model interpretation methods
* Understanding the limitations of traditional model interpretation methods
* Studying intrinsically interpretable (white-box) models
* Recognizing the trade-off between performance and interpretability
* Discovering newer interpretable (glass-box) models

### Chapter 04, Global Model-Agnostic Interpretation Methods
In the first part of this book, we introduced the concepts, challenges, and purpose of machine learning interpretation. This chapter kicks off the second part, which dives into a vast array of methods that are used to diagnose models and understand their underlying data. One of the biggest questions answered by interpretation methods is: What matters most to the model and how does it matter? Interpretation methods can shed light on the overall importance of features and how they—individually or combined—impact a model’s outcome. This chapter will provide a theoretical and practical foundation to approach these questions.
Initially, we will explore the notion of feature importance by examining the model’s inherent parameters. Following that, we will study how to employ permutation feature importance in a model-agnostic manner to effectively, reliably, and autonomously rank features. Finally, we will outline how SHapley Additive exPlanations (SHAP) can rectify some of the shortcomings of permutation feature importance.
This chapter will look at several ways to visualize global explanations, such as SHAP’s bar and beeswarm plots, and then dive into feature-specific visualizations like Partial Dependence Plots (PDP) and Accumulated Local Effect (ALE) plots. Lastly, feature interactions can enrich explanations because features often team up, so we will discuss 2-dimensional PDP and ALE plots.
The following are the main topics we are going to cover in this chapter:
* What is feature importance?
* Gauging feature importance with model-agnostic methods
* Using SHAP, PDP, and ALE plots to visualize:
  * Global explanations
  * Feature summary explanations
  * Feature interactions

### Chapter 05, Local Model-Agnostic Interpretation Methods
In the previous two chapters, we dealt exclusively with global interpretation methods. This chapter will foray into local interpretation methods, which are there to explain why a single prediction or a group of predictions was made. It will cover how to leverage SHapley Additive exPlanations (SHAP’s) KernelExplainer and also another method called Local Interpretable Model-agnostic Explanations (LIME) for local interpretations. We will also explore how to use these methods with both tabular and text data.
These are the main topics we are going to cover in this chapter:
* Leveraging SHAP’s KernelExplainer for local interpretations with SHAP values
* Employing LIME
* Using LIME for Natural Language Processing (NLP)
* Trying SHAP for NLP
* Comparing SHAP with LIME

### Chapter 06, Anchors and Counterfactual Explanations
In previous chapters, we learned how to attribute model decisions to features and their interactions with state-of-the-art global and local model interpretation methods. However, the decision boundaries are not always easy to define or interpret with these methods. Wouldn’t it be nice to be able to derive human-interpretable rules from model interpretation methods? In this chapter, we will cover a few human-interpretable, local, classification-only model interpretation methods. We will first learn how to use scoped rules called anchors to explain complex models with statements such as if X conditions are met, then Y is the outcome. Then, we will explore counterfactual explanations that follow the form if Z conditions aren’t met, then Y is not the outcome.
These are the main topics we are going to cover in this chapter:
* Understanding anchor explanations
* Exploring counterfactual explanations

### Chapter 07, Visualizing Convolutional Neural Networks
Up to this point, we have only dealt with tabular data and, briefly, text data, in Chapter 5, Local Model-Agnostic Interpretation Methods. This chapter will exclusively explore interpretation methods that work with images and, in particular, with the Convolutional Neural Network (CNN) models that train image classifiers. Typically, deep learning models are regarded as the epitome of black box models. However, one of the benefits of a CNN is how easily it lends itself to visualization, so we can not only visualize outcomes but also every step of the learning process with activations. The possibility of interpreting these steps is rare among so-called black box models. Once we have grasped how CNNs learn, we will study how to use state-of-the-art gradient-based attribution methods, such as saliency maps and Grad-CAM to debug class attribution. Lastly, we will extend our attribution debugging know-how with perturbation-based attribution methods such as occlusion sensitivity and KernelSHAP.
These are the main topics we are going to cover:
* Assessing the CNN classifier with traditional interpretation methods
* Visualizing the learning process with an activation-based method
* Evaluating misclassifications with gradient-based attribution methods
* Understanding classifications with perturbation-based attribution methods

### Chapter 08, Interpreting NLP Transformers
In the last chapter, we learned about applying explanation methods to a specific type of deep learning model architecture, convolutional neural networks. In this chapter, we will provide some tools to do the same with the transformer model architecture. Transformer models are becoming increasingly popular, and their most common use case is Natural Language Processing (NLP). We broached the subject of NLP in Chapter 5, Local Model-Agnostic Interpretation Methods. In this chapter, we will do so too but with transformer-specific methods and tools. First, we will discuss how to visualize attention mechanisms, followed by interpreting integrated gradient attributions, and lastly, exploring the Swiss Army knife that is the Learning Interpretability Tool (LIT).
These are the main topics we will cover:
* Visualizing attention with BertViz
* Interpreting token attributions with integrated gradients
* LIME, counterfactuals, and other possibilities with the LIT

### Chapter 09, Interpretation Methods for Multivariate Forecasting and Sensitivity Analysis
Throughout this book, we have learned about various methods we can use to interpret supervised learning models. They can be quite effective at assessing models while also uncovering their most influential predictors and their hidden interactions. But as the term supervised learning suggests, these methods can only leverage known samples and permutations based on these known samples’ distributions. However, when these samples represent the past, things can get tricky! As the Nobel laureate in physics Niels Bohr famously quipped, “Prediction is very difficult, especially if it’s about the future.”
Indeed, when you see data points fluctuating in a time series, they may appear to be rhythmically dancing in a predictable pattern – at least in the best-case scenarios. Like a dancer moving to a beat, every repetitive movement (or frequency) can be attributed to seasonal patterns, while a gradual change in volume (or amplitude) is attributed to an equally predictable trend. The dance is inevitably misleading because there are always missing pieces of the puzzle that slightly shift the data points, such as a delay in a supplier’s supply chain causing an unexpected dent in today’s sales figures. To make matters worse, there are also unforeseen catastrophic once-in-a-decade, once-in-a-generation, or simply once-ever events that can radically make the somewhat understood movement of a time series unrecognizable, similar to a ballroom dancer having a seizure. For instance, in 2020, sales forecasts everywhere, either for better or worse, were rendered useless by COVID-19!
We could call this an extreme outlier event, but we must recognize that models weren’t built to predict these momentous events because they were trained on almost entirely likely occurrences. Not predicting these unlikely yet most consequential events is why we shouldn’t place so much trust in forecasting models to begin with, especially without discussing certainty or confidence bounds.
This chapter will examine a multivariate forecasting problem with Long Short-Term Memory (LSTM) models. We will first assess the models with traditional interpretation methods, followed by the Integrated Gradient method we learned about in Chapter 7, Visualizing Convolutional Neural Networks, to generate our model’s local attributions.
But more importantly, we will understand the LSTM’s learning process and limitations better. We will then employ a prediction approximator method and SHAP’s KernelExplainer for both global and local interpretation. Lastly, forecasting and uncertainty are intrinsically linked, and sensitivity analysis is a family of methods designed to measure the uncertainty of the model’s output in relation to its input, so it’s very useful in forecasting scenarios. We will also study two such methods: Morris for factor prioritization and Sobol for factor fixing, which involves cost sensitivity.
The following are the main topics we are going to cover:
* Assessing time series models with traditional interpretation methods
* Generating LSTM attributions with integrated gradients
* Computing global and local attributions with SHAP’s KernelExplainer
* Identifying influential features with factor prioritization
* Quantifying uncertainty and cost sensitivity with factor fixing

### Chapter 10, Feature Selection and Engineering for Interpretability
In the first three chapters, we discussed how complexity hinders Machine Learning (ML) interpretability. There’s a trade-off because you may need some complexity to maximize predictive performance, yet not to the extent that you cannot rely on the model to satisfy the tenets of interpretability: fairness, accountability, and transparency. This chapter is the first of four focused on how to tune for interpretability. One of the easiest ways to improve interpretability is through feature selection. It has many benefits, such as faster training and making the model easier to interpret. But if these two reasons don’t convince you, perhaps another one will.
A common misunderstanding is that complex models can self-select features and perform well nonetheless, so why even bother to select features? Yes, many model classes have mechanisms that can take care of useless features, but they aren’t perfect. And the potential for overfitting increases with each one that remains. Overfitted models aren’t reliable, even if they are more accurate. So, while employing model mechanisms such as regularization is still highly recommended to avoid overfitting, feature selection is still useful.
In this chapter, we will comprehend how irrelevant features adversely weigh on the outcome of a model and thus, the importance of feature selection for model interpretability. Then, we will review filter-based feature selection methods such as Spearman’s correlation and learn about embedded methods such as LASSO and ridge regression. Then, we will discover wrapper methods such as sequential feature selection, and hybrid ones such as Recursive Feature Elimination (RFE). Lastly, even though feature engineering is typically conducted before selection, there’s value in exploring feature engineering for many reasons after the dust has settled and features have been selected.
These are the main topics we are going to cover in this chapter:
* Understanding the effect of irrelevant features
* Reviewing filter-based feature selection methods
* Exploring embedded feature selection methods
* Discovering wrapper, hybrid, and advanced feature selection methods
* Considering feature engineering

### Chapter 11, Bias Mitigation and Causal Inference Methods
In Chapter 6, Anchors and Counterfactual Explanations, we examined fairness and its connection to decision-making but limited to post hoc model interpretation methods. In Chapter 10, Feature Selection and Engineering for Interpretability, we broached the topic of cost-sensitivity, which often relates to balance or fairness. In this chapter, we will engage with methods that will balance data and tune models for fairness.
With a credit card default dataset, we will learn how to leverage target visualizers such as class balance to detect undesired bias, then how to reduce it via preprocessing methods such as reweighting and disparate impact remover for in-processing and equalized odds for post-processing. Extending from the topics of Chapter 6, Anchors and Counterfactual Explanations, and Chapter 10, Feature Selection and Engineering for Interpretability, we will also study how policy decisions can have unexpected, counterintuitive, or detrimental effects. A decision, in the context of hypothesis testing, is called a treatment. For many decision-making scenarios, it is critical to estimate their effect and make sure this estimate is reliable.
Therefore, we will hypothesize treatments for reducing credit card default for the most vulnerable populations and leverage causal modeling to determine its Average Treatment Effects (ATE) and Conditional Average Treatment Effects (CATE). Finally, we will test causal assumptions and the robustness of estimates using a variety of methods.
These are the main topics we are going to cover:
* Detecting bias
* Mitigating bias
* Creating a causal model
* Understanding heterogeneous treatment effects
* Testing estimate robustness

### Chapter 12, Monotonic Constraints and Model Tuning for Interpretability
Most model classes have hyperparameters that can be tuned for faster execution speed, increasing predictive performance, and reducing overfitting. One way of reducing overfitting is by introducing regularization into the model training. In Chapter 3, Interpretation Challenges, we called regularization a remedial interpretability property, which reduces complexity with a penalty or limitation that forces the model to learn sparser representations of the inputs. Regularized models generalize better, which is why it is highly recommended to tune models with regularization to avoid overfitting to the training data. As a side effect, regularized models tend to have fewer features and interactions, making the model easier to interpret—less noise means a clearer signal!
And even though there are many hyperparameters, we will only focus on those that improve interpretability by controlling overfitting. Also, to a certain extent, we will revisit bias mitigation through the class imbalance-related hyperparameters explored in previous chapters.
Chapter 2, Key Concepts of Interpretability, explained three model properties that impact interpretability: non-linearity, interactivity, and non-monotonicity. Left to its own devices, a model can learn some spurious and counterintuitive non-linearities and interactivities. As discussed in Chapter 10, Feature Selection and Engineering for Interpretability, guardrails can be placed to prevent this through careful feature engineering. However, what can we do to place guardrails for monotonicity? In this chapter, we will learn how to do just this with monotonic constraints. And just as monotonic constraints can be the model counterpart to feature engineering, regularization can be the model counterpart to the feature selection methods we covered in Chapter 10!
These are the main topics we are going to cover in this chapter:
* Placing guardrails with feature engineering
* Tuning models for interpretability
* Implementing model constraints

### Chapter 13, Adversarial Robustness
Machine learning interpretation has many concerns, ranging from knowledge discovery to high-stakes ones with tangible ethical implications, like the fairness issues examined in the last two chapters. In this chapter, we will direct our attention to concerns involving reliability, safety, and security.
As we realized using the contrastive explanation method in Chapter 7, Visualizing Convolutional Neural Networks, we can easily trick an image classifier into making embarrassingly false predictions. This ability can have serious ramifications. For instance, a perpetrator can place a black sticker on a yield sign, and while most drivers would still recognize this as a yield sign, a self-driving car may no longer recognize it and, as a result, crash. A bank robber could wear a cooling suit designed to trick the bank vault’s thermal imaging system, and while any human would notice it, the imaging system would fail to do so.
The risk is not limited to sophisticated image classifiers. Other models can be tricked! The counterfactual examples produced in Chapter 6, Anchors and Counterfactual Explanations, are like adversarial examples except with the goal of deception. An attacker could leverage any misclassification example, straddling the decision boundary adversarially. For instance, a spammer could realize that adjusting some email attributes increases the likelihood of circumventing spam filters.
Complex models are more vulnerable to adversarial attacks. So why would we trust them?! We can certainly make them more foolproof, and that’s what adversarial robustness entails. An adversary can purposely thwart a model in many ways, but we will focus on evasion attacks and briefly explain other forms of attacks. Then we will explain two defense methods: spatial smoothing preprocessing and adversarial training. Lastly, we will demonstrate one robustness evaluation method.
These are the main topics we will cover:
* Learning about evasion attacks
* Defending against targeted attacks with preprocessing
* Shielding against any evasion attack through adversarial training of a robust classifier

### Chapter 14, What’s Next for Machine Learning Interpretability?
Over the last thirteen chapters, we have explored the field of Machine Learning (ML) interpretability. As stated in the preface, it’s a broad area of research, most of which hasn’t even left the lab and become widely used yet, and this book has no intention of covering absolutely all of it. Instead, the objective is to present various interpretability tools in sufficient depth to be useful as a starting point for beginners and even complement the knowledge of more advanced readers. This chapter will summarize what we’ve learned in the context of the ecosystem of ML interpretability methods, and then speculate on what’s to come next!
These are the main topics we are going to cover in this chapter:
* Understanding the current landscape of ML interpretability
* Speculating on the future of ML interpretability



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


## Errata
*  [Page 4 - Understanding a simple weight prediction model](https://github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python-2E/blob/main/README.md#page-4---understanding-a-simple-weight-prediction-model)
*  [Page 27 - Interpretation method types and scopes] ()


## Page 4 - Understanding a simple weight prediction model
In Page 4 after the output line, I have written "on average, for every additional pound, there are 3.4 inches of height" but it should have been "on average, for every additional inch in height there are 3.4 additional pounds in weight"

## Page 27 - Interpretation method types and scopes
In Page 27 1st Pagragraph, I have written "odds of a rival team winning the championship today, the positive case would be that they own" but I should have written **won** not **own**
  






