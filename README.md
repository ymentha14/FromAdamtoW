# From-Adam-to-W

Compare performance in terms of accuracy, stability, and speed of convergence of three optimizers: SGD with momentum and weight decay, Adam and AdamW.

## Report

The technical report is accessible here: [report.pdf](https://github.com/ymentha14/FromAdamtoW/blob/master/report/report.pdf)

## Introduction

We run the three different optimization on three different tasks, namely text images, text classification and text speech recognition. 

Particular effort has been paid to the modularity of the code and
to achieve unbiased and reproducible results.

## Usage

To install the packages:

```
pip install -r requirements.txt
```

To run the code: 
```
python main.py --verbose
```

List of all arguments:

```
python main.py --help
```

- `--task_name` accept `text_cls`, `speech_cls` and `images_cls` for running a single experiment. Default is `all`.

- `--optimizer` accept `Adam`, `AdamW` and `SGD`. Default is `all`.

- `--num_runs`: repeat the final experiment `num_runs` times. Default is 5.

- `--grid_search`: when set, run the grid search and find the best hyper-parameters for every task. 

-  `--verbose`: print useful information during training. Default False.

- `--sample_size`: provide an int to limit the size of the dataset. Useful to speed up the training execution. By default, it uses all datasets.

- `--train_size_ratio`: percentage of data to use for training. Default is `0.8`.

- `--batch_size`: the size of the batch used during training. Default is 32.

- `--seed`: the random seed used for results reproducibility. Default is 42.

- `--patience`: grid_search make uses of early stopping to accelerate the training time. The patience is the number of epoches since there are no more improvement in the loss validation before terminating the training. 


- `--k` number of times to repeat the cross validation.

- `--overwrite_best_param`: specify this option when running `grid_serach`.
If this option is set to true, the best parameter found during the cross-validation phase is stored in the `best_{task_name}.json` file, so that such hyper parameters can be used during the actual experiment. 

## Contents

This code aims to run three experiments (one for each task), using
ideal hyper-parameters specific for every task. After having run all experiments, we store the results in a log file `log/log_{task_name}.json`.

We found such scheduling to be convenient:

1) Run a large grid search to find the best hyper-parameters for every task.
2) With the best hyper-parameters found, run the experiments.
3) Analyze the results.

### Models

The models are implemented in the folder `tasks`. All override three methods
- `get_full_dataset()` returns the correct dataset.
- `get_model()` returns the correct PyTorch model.
- `get_scoring_function()` returns a function that scores the current model on 
some labeled data.

### Grid search

To perform the grid search, first, we divide the dataset into training
and test data. The test data is **never** used during the grid search phase, to respect the amount of information available during training.

To run the grid search, the class `Tester` in file `tester.py` is used.
The grid of hyper parameters to test are stored in `params/grid_search_{task}.json`.

All possible combinations of hyper-parameters are checked for every optimizer on the same model architecture. Note also that we searched for a good model architecture and sticked to that for the whole work.

We decided to use cross-validation to get a better estimation of the best 
hyper-parameters combination. Instead of using k-fold cross-validation, we divided randomly the training data into training and test data `k` times. This is also known as [repeated random sub-sampling validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics).

This choice has been made since, using a small `k` (like 3, out default choice due to time constraint), we would have remained with a very small train dataset, which would have been too small if compared with the one used during the testing phase, which hence might result in a grid search that finds the best hyper-parameters for a very different dataset then the one actually used during testing.

Since this process is very time consuming, the authors have already provided a
`best_{task}.json` fine-tuned after having run a grid of approximately 
30 different combinations, so that after having pulled the repo, you can directly jump to the more interesting testing phase.


### Extraction of best parameters

After having run the grid search, if the option `--overwrite_best_param` is true, then it is time to analyze the results to extract the best hyper-parameters to store in `best_{task}.json`.

We decided to take as best parameter combination the one that had the highest validation mean for the same epoch. Hence, as best parameters, we do not have only the hyper-parameters of the optimizer, but even the number of epochs we need to train our model for, which is very important information to know during the test phase.

Lastly, we decided to discard epochs that were not reached by all attempts, even when the accuracy for that epoch was the highest for such hyper-parameters combination. We assumed that not all observations reached such epoch (due to early stopping), then it is likely that results for that epoch are noisy and hence not meaningful.
 
### Testing Phase

Here, we first load the best hyper-parameters found during the grid search and 
then train `num_runs` times the same model and save the results. We train the model multiple times for having variance and more reliable conclusion.

## Dataset

### Text classification
[AG_NEWS dataset](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)

### Image classification
[MNIST dataset](http://yann.lecun.com/exdb/mnist/)
28x28 greyscale images of 60000 training handwritten digits from 0 to 9, with a test size of 10000 images.
The train and test split has been kept to 6/7, to resemble the original MNIST settings (60000 training images and 10000 validation ones).

### Speech classification
[Berlin Database of Emotional Speech](http://emodb.bilderbar.info/start.html)

## Contributors
Jonathan Besomi, Stefano Huber, Yann Mentha
