# From-Adam-to-W

## Introduction
This library's aim is to check the performance in terms of accuracy, stability and
speed of the algorithm AdamW if compared with standard SGD with momentum and Adam.

The three optimizers are run on three different tasks namely NLP, image processing
and speech recognition. 

With this code it is possible to run both the grid search in order to look for the
best hyper parameters for each specific optimizer, and to run the experiment with
such optimal parameters.

Particular effort has been paid to the modularity of the code and
to achieve unbiased and reproducible results.


## Usage

First you need to install all dependencies: after having cloned the repo
in a directory, create a Python3 virtual environment and run

```
pip install -r requirements.txt
``` 

Then, to run the code, simply run:
```
python main.py --verbose
```

Other options are available, to show them use
```python main.py --help```
Such options include:
- ```--task_name``` with three choices, ```text_cls```, ```speech_cls```, ```images_cls```
if you want to run a single experiment. By default, it runs all experiments.
- ```--optimizer``` with choices ```all```, ```Adam```, ```AdamW```, ```SGD```
 to test only one or all the possible optimizers. Default ```all```.
- ```--num_runs```: repeat the final experiment ```num_runs``` times,
in order to provide more accurate results (with variance). Default 5.
- ```--grid_search```: instead of running the experiment, run the grid search to
look for the best hyper parameters for every task. Default False
-  ```--verbose```: print useful information during training. Default False.
- ```--sample_size```: provide an int to limit the size of the dataset. Useful to 
speed up the training execution. By default it uses all dataset.
- ```--train_size_ratio```: percentage of data to use for training. 
By default ```0.8```.
- ```--batch_size```: the size of the batch used during training. By default 32.
- ```--seed```: the random seed. Specify it if you want to have random results, 
but consistent among different runs. By default ```42```.
- ```--patience```: we use early stopping to limit the time used to decree that
the model has converged while performing grid search, instead of waiting until we
have run all the epochs. 
Patience is the number of epochs to wait until no improvement is recorded. 
When ```patience``` epochs have elapsed, we assume the model has converged. 
By default 7.
- ```--k``` specify it only when running ```grid_search```:
we use k-cross validation, namely we repeat the experiment ```k``` times,
in order to have a more accurate estimate. Default: 3.
- ```--overwrite_best_param```: specify this option when running ```grid_serach```.
If this option is set to true, the best parameter found during the cross validation
phase are stored in the ```best_{task_name}.json``` file, so that such hyper 
parameters can be used during the actual experiment. 


## Contents

The aim of this code is to run three experiments (one for each task), using
ideal hyper parameters specific for every task. After having run such experiments,
we store the results achieved during the training phase in a log file 
`log/log_{task_name}.json`, so that we can extract valuable information like
the test accuracy and loss for every epoch of training, in order to analyze
statistically the figures using `notebooks/visu_stat.ipynb`, and come to a conclusion on
the superiority of AdamW over the other optimizers.

We found such scheduling to be convenient:
1) Run a large grid search to find the best hyper-parameters for every task.
2) With the best hyper parameters found, run the experiments.
3) Analyze the results.

Let us look deeper in the code to see how the three tasks are implemented
and how we managed to achieve modularity. 


### Code
```
├── data          # Data directory, data is downloaded only when needed
├── helper.py     
├── log           # Result logs, they are used when analyzing the results
│   ├── log_images_results.json
│   ├── log_text_results.json
│   └── log_speech_results.json
├── main.py      
├── params        # parameters for grid search or test with best hyper-parameters
│   ├── best_images.json
│   ├── best_speech.json
│   ├── best_text.json
│   ├── grid_search_images.json
│   ├── grid_search_speech.json
│   └── grid_search_text.json
├── pytorch_helper.py   
├── README.md
├── report        # Notebooks with code to plot stats on the results 
│   ├── Analyze Results.ipynb
│   └── log_images_results.json
├── Report_SER.ipynb
├── requirements.txt       # Requirement file for virtualenv
├── tasks         # Model implementation for every specific task 
│   ├── images_cls.py
│   ├── speech_cls.py
│   └── text_cls.py
├── tester.py    # Tester class, it runs cross validation or the testing experiments 
├── visualize.py
└── visu_stat.ipynb
```

### Models
The models are implemented in the folder `tasks`, and all override three methods
- `get_full_dataset()`: returns the correct dataset.
- `get_model()`: return the correct Pytorch model.
- `get_scoring_function()` returns a function that scores the current model on 
some labeled data.

### Grid search
In order to perform the grid search, first we divide the dataset into training
and test data. The test data is **never** used during the grid search phase, 
in order to respect the amount of information available during training.
To run the grid search, the class `Tester` in file `tester.py` is used.
The grid of hyper parameters to test are stored in `params/grid_search_{task}.json`.
All possible combinations of hyper parameters are checked for every optimizer, 
without changing the model architecture (otherwise, difference in results may be 
due to model differences rather than difference in the optimizers).

We decided to use cross validation to get a better estimation of the best 
hyper-parameters combination: but, instead of using k-fold cross validation,
we divided randomly the training data into training and test data for `k` times, but using 
using a fixed ratio of 0.8 (instead of a ratio dependent from `k`,
as would have been the case for the k-fold cross validation). This choice has been 
made since, using a small `k` (like 3, out default choice due to time constraint),
we would have remained with a very small train dataset, which would have been too
small if compared with the one used during the testing phase, which hence might
result in a grid search that finds the best hyper parameters for a very different
dataset then the one actually used during testing (since it is considerably bigger).

When the option `--overwrite_best_param` is specified, at the end of the grid search 
the best hyper-parameters for the specific task are updated in the file 
`best_{task}.json`. In this way, when later we run the actual experiment, we can 
load the best set of hyper parameters for every task and use it as setup option
for the three optimizers. 

Since this process is very time consuming, the authors have already provided a
`best_{task}.json` fine tuned after having run a grid of approximately 
30 different combinations, so that after having pulled the repo, you can directly
jump to the more interesting testing phase.


### Extraction of best parameters
After having run the grid search, if the option `--overwrite_best_param` is true,
then it is time to analyze the results in order to 
extract the best hyper parameters to store in `best_{task}.json`.
We do it using the helper `compute_best_parameter()`.
This function receives for every optimizer a list of 
observations for the validation accuracies among the various epochs of training.

We decided to take as best parameter combination the one that had the highest 
validation mean for the same epoch. Hence, as best parameter, we do not have only
the hyper parameters of the optimizer, but even the number of epochs we need to train 
our model for, which is a very important information to know during the test phase.

Lastly we decided to discard epochs that were not reached by all attempts, even if 
the accuracy for that epoch was the highest for such hyper-parameters combination.
This choice was done since we believe that if not all observations reached such epoch
(due to early stopping), then it is likely that results for that epoch are noisy
and hence not meaningful.
 


### Testing Phase

This is the interesting part of the project, and the one you would execute by
running `python main.py`: we want to record the train and test 
accuracy across the various epochs to verify whether it is the case that
AdamW converges faster or to better solutions then Adam, using SGD as baseline
of comparison (since it is probably the most known and used optimizer).

To do so, first we load the best hyper-parameters found during the grid search,
then we train `num_runs` times the same model logging the statistics. 

We train the model multiple times in order to be able to have estimates 
provided with variance to get to a more reliable conclusion during 
the analysis phase.

 

## Dataset description
###Text classification
Toxic Comment Classification Challenge

###Image processing
MNIST dataset: 28x28 greyscale images of 60000 training hand written digits from
0 to 9, with a test size of 10000 images. The accuracy is computed easily
by counting the number of correct predictions over the total amount of predictions.

###Speech recognition



- Study the relation with EarlyStopping ?



**TODO**

- [ ] Scheduler https://pytorch.org/docs/master/_modules/torch/optim/lr_scheduler.html#StepLR, study/understand scheduler.  

- [ ] Automatically download wav files in the correct folder (Yann?)

- [ ] Best params file should be created automatically after CV. For now, we need to look at the results and pick the final best params. 
- [ ] Global get_scoring_function ?
- [ ] Add sample size under log


## implementation

- Weights initialization: kept PyTorch default, Most layers are initialized using Kaiming Uniform method. Example layers include Linear, Conv2d, RNN etc. If you are using other layers, you should look up that layer on this doc. If it says weights are initialized using U(...) then its Kaiming Uniform method.
--> say want to test in practice SDG vs. Adam vs. AdamW on most common scenario, where we assume practitioners does not change other default settings.

One of Adagrad's main benefits is that it eliminates the need to manually tune the learning rate, since it adapts naturally over time. In fact, most implementations use a default value of 0.01.


- "Pytorch, developed by Facebook AI Research and a team of collaborators, allows for rapid iteration and debugging"

- Use of PyTorch: flexible and clear API, full control of the code, very modular.

- Flexible structure ...

- Didn't use any extra tools such as Pytorch-lightning or Skorch as they even super useful for certain task as they abstract and put the complexity away, they give less control over the code.

According to Geoff Hinton: "Early stopping (is) beautiful free lunch"


Our implementation Code: almost pure PyTorch, see the details on the README attached code. Emphasis on reproducibility, should be possible to produce the same exact results just by running the main file without arguments. \footnote{\texttt{python main.py}}




## Contributors
Jonathan Besomi, Stefano Huber, Yann Mentha
