# From-Adam-to-W

Performance improvement of AdamW on Adam optmizers on controlled dataset


**USAGE**

First you need to install all dependencies: in a virtual environment run

```
pip install -r requirements.txt
``` 

Then, to run the code, simply run:
```
python main.py --verbose
```

Python version required >= 3.6

**TODO**

- Early stopping ?
- Scheduler ? https://pytorch.org/docs/master/_modules/torch/optim/lr_scheduler.html#StepLR


**Text classification**

Dataset:

- Toxic Comment Classification Challenge

**IDEA**

- Study the relation with EarlyStopping ?
command used : python3 main.py --params_file=./params/params.json --task_name=speech_cls --cross_validation --verbose --num_epochs=1
