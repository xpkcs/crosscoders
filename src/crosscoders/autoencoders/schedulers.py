

import torch

from crosscoders.config import Config, get_config

CONFIG: Config = get_config()




def get_scheduler_lr(optimizer, total_steps=CONFIG.batch.n_batches, decay_start_fraction=0.8):

    def lr_linear_tail(current_step):

        decay_start_step = int(total_steps * decay_start_fraction)
        decay_value      = 1.
        decay_progress   = 0. \
            if current_step < decay_start_step \
            else (current_step - decay_start_step) / (total_steps - decay_start_step)

        return decay_value - decay_progress


    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_linear_tail)


def get_scheduler_lambda_s(lambda_s_max, total_steps=CONFIG.batch.n_batches):

    def lambda_s_linear(current_step):

        return current_step / (total_steps - 1)


    class Scheduler:

        def __init__(self):
            self.current_step = 0 # int(0.1 * total_steps) # 0
            self.step()

        def step(self):
            self.lambda_s = lambda_s_max * lambda_s_linear(self.current_step)
            self.current_step += 1

        def get_lambda_s(self):
            return self.lambda_s


    return Scheduler()