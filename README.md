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

- [ ] Scheduler https://pytorch.org/docs/master/_modules/torch/optim/lr_scheduler.html#StepLR, study/understand scheduler.  

- [ ] Automatically download wav files in the correct folder (Yann?)

- [ ] Best params file should be created automatically after CV. For now, we need to look at the results and pick the final best params. 
- [ ] Global get_scoring_function ?

**Text classification**

Dataset:

- Toxic Comment Classification Challenge

**IDEA**

- Study the relation with EarlyStopping ?

**Command**

```bash

[ ] python3 main.py --task_name speech_cls --grid_search --verbose --num_epochs=1

```


**VERIFY the following**

- [ ] _optimizer name_s should be 'Adam', 'AdamW' and 'SGD'.
