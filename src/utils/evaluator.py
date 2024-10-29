from types import FunctionType

class Evaluator():
    def __init__(self, early_stoping_step:int) -> None:
        self.best_run = {}
        self.epoch = 1
        self.best_epoch = 1
        self.accuracy = {}
        self.early_stoping_step = early_stoping_step
        
    def run_evaluation(self, evaluation_name: str, eval_fnct: FunctionType, inputs: list):
        accuracy = eval_fnct(*inputs).item()
        if evaluation_name not in self.accuracy:
            self.accuracy[evaluation_name] = accuracy
        else:
            self.accuracy[evaluation_name] += accuracy
        
    def reset(self):
        self.epoch += 1
        self.accuracy = {}
        
    def check_early_stoping(self, current_epoch: int):
        if self.early_stoping_step < 0:
            return False
        return current_epoch >= self.best_epoch + self.early_stoping_step
        
    
    def step(self, evaluation_name:str, set_lenght:int, check_for_best_epoch:bool=True):
        self.accuracy[evaluation_name] /= set_lenght
        if check_for_best_epoch and (evaluation_name not in self.best_run or self.accuracy[evaluation_name] > self.best_run[evaluation_name]):
            self.best_run[evaluation_name] = self.accuracy[evaluation_name]
            self.best_epoch = self.epoch
            return True, self.best_run[evaluation_name]
        return False, self.accuracy[evaluation_name]