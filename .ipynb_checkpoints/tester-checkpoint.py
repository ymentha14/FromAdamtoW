import logging
import time
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from copy import copy
import helper as h
import json
from pathlib import Path
from IPython.core.debugger import set_trace



class Tester:
    """
    A tester needs to be run for one task: that is, for each dataset,optimizer and hyperparameter combination.
    """

    def __init__(self,args,task_data,task_model,optim,param):
        """Initialize the class Tester
        
        Args:
            args: parsed arguments
            task_data: Dataloader to the dataset
            task_model: model constructor for the network that performs decently on the current dataset
            optim: torch optimizer used
            param: dict of parameters for the model/dataset/optimizer combination
        """

        self.args = args
        self.task_data = task_data
        self.model_constructor = task_model
        self.optim = optim
        self.param = param

    def train(self):
        """
        Perform one training on the given inputs and return the elapsed time.
        """
        if self.args.verbose:
            print("Start training ...")

        start_time = time.time()

        # 1. Construct again the model
        self.model = self.model_constructor()

        # 3. Effectively train the model
        self._run_all_epochs()

        # 4. Store the time
        end_time = time.time()
        train_time = end_time - start_time

        if self.args.verbose:
            print("Finish training... after {:.2f}s".format(train_time))
        self.train_time = train_time

        return train_time
    
    def log(self,log_path):
        """append the scores of the current run to the json in log_path"""
        
        log_path_posix = Path(log_path)
        if not log_path_posix.exists():
            with open(log_path,'w') as f: 
                json.dump({}, f, indent=4) 
        date = datetime.now().strftime("%m_%d_%y-%H_%M")
        log_data = copy(self.param)
        log_data['optim'] = str(self.optim)
        log_data['losses'] = self.losses
        log_data['train_time'] = self.train_time
        new_data = {date:log_data}
        #old_log = json.loads(log_path)
        
        with open(log_path,'r') as f: 
            old_log = json.load(f)
        
        old_log.update(new_data)
        with open(log_path,'w') as f: 
            json.dump(old_log, f, indent=4) 
        

    def _run_all_epochs(self):
        """
        run the current model over the number of epochs specified in the args.
        """
        # 100 by default
        num_epochs = self.args.num_epochs
        # hard-coded criterion since we only use cross-entropy loss
        criterion = nn.CrossEntropyLoss()
        
        optimizer = self.optim(self.model.parameters(),**h.adapt_params(self.param))
        self.losses = []
        self.f1s = []
        for epoch in range(num_epochs):
            
            for X_batch,y_batch in self.task_data:
                output_batch = self.model(X_batch)
                loss = criterion(output_batch,y_batch)
                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                self.losses.append(loss.item())
                #TODO compute F1 score in here
                #f1 = ...
                #f1s.append(f1)
                

            if self.args.verbose:
                print(
                    "({}/{}) Training loss: {:.3f}".format(
                        epoch + 1, num_epochs, loss
                    ),
                    end="\r" if epoch + 1 != num_epochs else "\n",
                )
        print("=" * 60 + f"\nSuccess: final loss {loss}")
        
    def run(self):
        return self.train()
