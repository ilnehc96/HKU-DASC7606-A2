# HKU-DASC7606-A2
HKU DASC-7606 Assignment 2 NLP: Machine Reading Comprehension

**For questions and discussion**,
- We encourage you to use [GitHub Issues](https://github.com/ilnehc96/HKU-DASC7606-A2/issues) of this repository.
- Or if you prefer online doc: [Discussion doc](https://connecthkuhk-my.sharepoint.com/:w:/g/personal/ilnehc_connect_hku_hk/EXP4PuxeMPZFtuZwbhkltLcBd55Un8CENtFojxV60JU-Gw?e=AapZoz).

This codebase is only for HKU DASC 7606 (2022-2023) course. Please don't upload your answers or this codebase to any public platforms (e.g., GitHub) before permitted. All rights reserved.


## 1 Introduction

### 1.1 What is Machine Reading Comprehension?
Open-domain Question Answering (ODQA) is a type of natural language processing task, which asks a model to produce answers to factoid questions posed in natural language. Open-domain means that the question is not limited to a specific domain but could be anything about real-world knowledge and common sense. Based on whether external context and background are provided to help answer the question, there are two types of question answering, namely **Open-book QA** and **Closed-book QA**. In Open-book QA, the system is equipped with an abundant knowledge base to refer to, while a Closed-book QA system has to memorize all the world knowledge in its parameter. Usually, a standard Open-book QA system works in a two-step pipeline: First, a retriever scans the external knowledge base to filter related context, and then a reader is responsible for extracting or generating the answer according to the retrieved context. In this assignment, we focus on a simpler case in Open-book QA where the relevant document is provided. It is also called **Machine Reading Comprehension (MRC)** in some literature.

### 1.2 What will you learn from this assignment?
This assignment will walk you through specific open-domain question-answering tasks. You can refer to the following example for an intuitive illustration.

    Context: The Review of Politics was founded in 1939 by Gurian, modeled after German Catholic journals. It quickly emerged as part of an international Catholic intellectual revival, offering an alternative vision to positivist philosophy. For 44 years, the Review was edited by Gurian, Matthew Fitzsimons, Frederick Crosson, and Thomas Stritch. Intellectual leaders included Gurian, Jacques Maritain, Frank O'Malley, Leo Richard Ward, F. A. Hermens, and John U. Nef. It became a major forum for political ideas and modern political concerns, especially from a Catholic and scholastic tradition.

    Question: Over how many years did Gurian edit the Review of Politics at Notre Dame?

    Answer: 44

Besides, you will fine-tune a pre-trained language model into Question Answering task from scratch, and you will also be asked to utilize the standard metrics for evaluation.

The goals of this assignment are as follows:

- Understand the problem formulation and the challenge of open-domain question answering.

- Understand how pre-trained language models can be used for extractive open-domain question answering.

- Implement a Question Answering (Reading Comprehension) model by fine-tuning a pre-trained language model from scratch and utilizing existing metrics for evaluation.

## 2 Setup

You can work on the assignment in one of two ways: locally on your own machine, or on a virtual machine on HKU GPU Farm.

### 2.1 Working remotely on HKU GPU Farm (Recommended)

Note: after following these instructions, make sure you go to work on the assignment below (i.e., you can skip the Working locally section).

As part of this course, you can use HKU GPU Farm for your assignments. We recommend you follow the quickstart provided by the official website to get familiar with HKU GPU Farm.

After checking the quickstart document, make sure you have gained the following skills:

+ Knowing how to access the GPU Farm and use GPUs in interactive mode. We recommend using GPU support for this assignment, since your training will go much, much faster.
+ Getting familiar with running Jupyter Lab without starting a web browser.
+ Knowing how to use tmux for unstable network connections.

### 2.2 Working locally

If you have the GPU resources on your own PC/laptop. Here’s how you install the necessary dependencies:

**Installing GPU drivers (Recommend if working locally)**: If you choose to work locally, you are at no disadvantage for the first parts of the assignment. Still, having a GPU will be a significant advantage. If you have your own NVIDIA GPU, however, and wish to use that, that’s fine – you’ll need to install the drivers for your GPU, install CUDA, install cuDNN, and then install PyTorch. You could theoretically do the entire assignment with no GPUs, though this will make training the model much slower.

**Installing Python 3.8+**: To use python3, make sure to install version 3.8+ on your local machine.

**Virtual environment**: If you decide to work locally, we recommend using a virtual environment via anaconda for the project. If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed globally on your machine. To set up a conda virtual environment, run the following:
        
        conda create -n qa_env python=3.8
        conda activate qa_env

Install the PyTorch environment following the official instructions. Here we use PyTorch 1.10.1 and CUDA 11.3. You may also switch to another version by specifying the version number:

        pip install torch==1.10.1+cu113  -f https://download.pytorch.org/whl/torch_stable.html

Install pytorch-transformers, numpy and tqdm:

        pip install pytorch-transformers==1.2.0, numpy, tqdm





## 3 Working on the assignment

### 3.1 Basis of pre-trained language models
Basic knowledge about the language model pre-training is necessary for completing the task. Please refer to the related chapter ([15.8](https://d2l.ai/chapter_natural-language-processing-pretraining/bert.html)) in *Dive Into Deep Learning* and the blog written by [Jay Alammar](https://jalammar.github.io/illustrated-bert/) for a general introduction of pre-training techniques, how the BERT models work, and etc.

If you are interested, please read the related papers (e.g., [BERT](https://arxiv.org/abs/1810.04805) and [SpanBERT](https://arxiv.org/abs/1907.10529) ) for more details.


### 3.2 Task description
The datum of open-domain question answering is a triplet $(q, a, D)$, where $q,a$ are question and answer respectively, and $D$ is the external document and context. Open domain question answering aims to correctly answer $q$ by extracting $a$ from $D$. Note here we assume that the question $q$ is always answerable and $a$ is always a substring within $D$. Open-domain question answering has drawn increasing attention in recent years because of its wide application in natural language processing and vertical commerce setting.

In this assignment, you are going to fine-tune a pre-trained language model for the QA task with the provided skeleton code. You will be provided with a standard dataset. 
<!-- The dataset can be found in this [link](https://drive.google.com/file/d/1M1gPsfz0Ts60U8CifxB1_1Ki8AcPWljy/view?usp=sharing). The dataset file is named as train/valid/test.json.  -->
<!-- There are 78311 cases in the training set, 8510 cases in the validation set, and 20302 cases in the test set.  -->
<!-- More details about the data format can be found in function *preprocess_data* in the data.py which is provided in the skeleton code. -->


### 3.3 Get Code

The code structure is as follows:

```
└── src
    ├── dataset
    │   ├── test.json               # test data (which will be released later)
    │   ├── train.json              # train data
    |   └── valid.json              # validation data
    ├── main.py                     # main file to train and evaluate the model
    ├── utils_squad.py              # utillity data structures and funcions
    ├── utils_squad_evaluate.py     # official SQuAD v2.0 evaluation script
    └── README.md
```

#### Train

```
python main.py --train_file dataset/train.json --predict_file dataset/valid.json --model_type bert --model_name_or_path bert-base-uncased  --output_dir output/ --version_2_with_negative --do_train --do_eval  --do_lower_case --overwrite_output --save_steps 0
```

Ps. Note that we are using SQuAD version 2.0, `--version_2_with_negative` argument is necessary. Please find descriptions and functions of arguments in the script.

#### Evaluation

```
python main.py --train_file dataset/train.json --predict_file dataset/test.json --model_type bert --model_name_or_path output/  --output_dir output/eval/ --version_2_with_negative --do_eval --do_lower_case
```

Ps. The argument `model_name_or_path` here should be the ourput saving directory in training to load saved checkpoint, and `output_dir` is the directory to save evaluation results.



### 3.4 Assignment tasks

**Task 1: Implement the self-attention function**

**Task 2: Implement the residual connection**

**Task 3: Implement the feed-forward layer**

We have provided an almost-complete code for the BERT model. Please complete the code in the places tagged with  `Write Your Code Here` in [modeling_bert.py](https://github.com/ilnehc96/HKU-DASC7606-A2/blob/main/modeling_bert.py).


**Task 4: Implement the data preprocessing function**

This function is responsible for converting the raw dataset into the required format for training the model. You will need to understand the preprocessing steps needed for the Machine Reading Comprehension task and use them to complete the code in the places tagged with `Write Your Code Here` in [utils_squad.py](https://github.com/ilnehc96/HKU-DASC7606-A2/blob/main/utils_squad.py).

**Task 5: Implement the training pipeline**

**Task 6: Implement the validation pipeline**

We have provided an almost-complete code for fine-tuning and evaluating the BERT on the QA task. Please complete the code in the places tagged with  `Write Your Code Here` in [main.py](https://github.com/ilnehc96/HKU-DASC7606-A2/blob/main/main.py) to adapt BERT for our own interest.

**Task 7: Predict the outputs of the test set**

This task requires you to predict the outputs of the test set which will be released 7 days before the deadline. To accomplish this task, you will need to make some modifications to the validation code. You are encouraged to use more advanced models and adopt some practical tricks in hyper-parameter tuning to improve model performance. 

**Task 8: Write a report (including the introduction, method, and experiment sections)**

Your report should include three main sections: introduction, method, and experiment. 

You are required to improve the baseline model with your own configuration. There are lots of ways to improve the baseline model. Here are some suggestions for you.

+ Hyper-parameter tuning.
There are lots of important hyper-parameters.
  + About optimizer: learning rate, batch size, warm-up and training epochs, etc.
  + About embedding: Add input type embedding to indicate different parts of inputs.

+ Different neural network architectures for predicting the span.
You may choose a more advanced neural network architecture and design your own customized neural network for predicting the answer span.

+ Loss functions designs.
You may add label smoothing or other tricks or come up with a new loss function or training objectives.

+ Others. There are also many other possible techniques to explore. For example, you can also design your own training strategies. Besides, you can also explore some PLM fine-tuning skills in this project.

In the method section, describe the approaches you used to improve the baseline model performance. In the experiment section, analyze the dataset statistics, qualitative evaluations of the model's performance, and case analysis. 





<!-- **Q5: Do something extra! (IMPORTANT)**

You are required to improve the baseline model with your own configuration. There are lots of ways to improve the baseline model. Here are some suggestions for you.

+ Hyper-parameter tuning.
There are lots of important hyper-parameters.
  + About optimizer: learning rate, batch size, warm-up and training epochs, etc.
  + About embedding: add input type embedding to indicate different parts of inputs.

+ Different neural network architectures for predicting the span.
You may choose a more advanced neural network architecture and design your own customized neural network for predicting the answer span.

+ Loss functions designs.
You may add label smoothing or other tricks or come up with a new loss function or training objectives.

+ Others. There are also many other possible techniques to explore. For example, you can also design your own training strategies. Besides, you can also explore some PLM fine-tuning skills in this project. -->


### 3.5 Files to submit (4 items in total)

1.  Prepare a final report in PDF format (no more than 4 pages)

    1.1 Introduction. Briefly introduce the task & background & related works.

    1.2 Methods. Describe what you did to improve the baseline model performance. For example, this may include but is not limited to: (i) Hyper-parameter tuning, e.g. learning rate, batch size, and training epochs. (ii) Different neural network architectures. (iii) Loss functions. 
    
    1.3 Experiments & Analysis **(IMPORTANT)** Analysis is the most important part of the report. Possible analysis may include but is not limited to (i) Dataset analysis (dataset statistics) (ii) Qualitative evaluations of your model. Select several specific cases in the dataset and see if your model correctly finds the answer. Failure case analysis is also suggested. (iii) Ablation studies. Analyze why better performance can be achieved when you made some modifications, e.g. hyper-parameters, model architectures, and loss functions. The performance on the validation set should be given to validate your claim.
2. Codes

    2.1 All the codes.
    
    2.2 README.txt that describes which python file has been added/modified.

3. Models

    3.1 Model checkpoint (i.e., pytorch_model.bin)
    
    <!-- 3.2 Model training log (log.txt) -->

4. Generated results on the test set. Please submit the predicted answer for each question in a single file (i.e., predictions_.json) 

If your student id is 12345, then the file should be organized as follows:

        12345.zip
        |-report.pdf
        |-BERTQA
        |   |-README.md
        |   |-your code
        |-log
        |   |-pytorch_model.bin
        |-results
        |   |-predictions_.json


### 3.6 When to submit?

The deadline is Apr 7 (Fri).

Late submission policy:

10% for late assignments submitted within 1 day late. 

20% for late assignments submitted within 2 days late.

50% for late assignments submitted within 7 days late.

100% for late assignments submitted after 7 days late.

### 3.7 Need More Support?

For any questions about the assignment which potentially are common to all students, your shall first look for related resources as follows,
- We encourage you to use [GitHub Issues](https://github.com/ilnehc96/HKU-DASC7606-A2/issues) of this repository.
- Or if you prefer online doc: [Discussion doc](https://connecthkuhk-my.sharepoint.com/:w:/g/personal/ilnehc_connect_hku_hk/EXP4PuxeMPZFtuZwbhkltLcBd55Un8CENtFojxV60JU-Gw?e=AapZoz).

For any other private questions, please contact Li Chen (ilnehc@connect.hku.hk) and Xueliang Zhao (xlzhao22@connect.hku.hk) via email.

## 4 Marking Scheme:

Marks will be given based on the performance that you achieve on the test and the submitted report file. TAs will perform an evaluation of the predicted answers.

The marking scheme has two parts, (1) the performance ranking based on F1 and EM (70% marks) and (2) the final report (30% marks):

1. For the performance part (70%), the mark will be given based on the performance (0.5 * EM + 0.5 * F1) of your model:

    (1) 0.5 * EM + 0.5 * F1 larger than 76 will get the full mark of this part.

    (2) 0.5 * EM + 0.5 * F1 between 72-76 will get 90% mark of this part.
    
    (3) 0.5 * EM + 0.5 * F1 between 68-72 will get 80% mark of this part.
    
    (4) 0.5 * EM + 0.5 * F1 between 64-68 will get 70% mark of this part.
    
    (5) 0.5 * EM + 0.5 * F1 between 60-64 will get 60% mark of this part.
    
    (6) 0.5 * EM + 0.5 * F1 larger than 10 will get 50% mark of this part.
    
    (7) Others will get 0% mark.


2. For the final report part (30%): The marks will be given mainly based on the richness of the experiments & analysis.

    (1) Rich experiments + detailed analysis: 90%-100% mark of this part.
    
    (2) Reasonable number of experiments + analysis: 70%-80% mark of this part.
    
    (3) Basic analysis: 50%-60% mark of this part.
    
    (4) Not sufficient analysis: lower than 50%.


## Reference

1. Dive Into Deep Learning https://d2l.ai/chapter_natural-language-processing-pretraining/bert.html
2. SpanBERT paper https://aclanthology.org/2020.tacl-1.5/
3. Lil's log https://lilianweng.github.io/posts/2020-10-29-odqa/
4. Jay Alammar's blog https://jalammar.github.io/illustrated-bert/

