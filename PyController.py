import os
from multiprocessing import Process

class py_process_pool():
    def __init__(self, pool_size):
        self.pool = []
        self.pool_size = pool_size
        self.tasks_num = 0
        self.tasks_done_num = 0
    
    # initialize a py process and add it to the pool
    def add_process(self, name):
        p = py_process(name)
        p.start()
        self.pool.append(p)
        self.tasks_num += 1
        print('add one process: {}'.format(name))
    
    def remove_dead_process(self):
        for p in self.pool:
            # process is dead
            if (p.is_alive() == False):
                print('remove one process: {}'.format(p.name))
                self.pool.remove(p)
                self.tasks_num -= 1
                self.tasks_done_num += 1

    
class py_process(Process):
    def __init__(self, name):
        super(py_process, self).__init__()
        self.name = name
    def run(self):
        os.system('python3 ' + str(self.name))

def task_schedule(task_name, seed_name):
    task_schedule = []
    for t_name in task_name:
        for s_name in seed_name:
            task_schedule.append((t_name, s_name))
    return task_schedule


if __name__ == '__main__':

    # define your rl envs
    task_name = ['asterix', 'breakout', 'freeway', 'seaquest', 'space_invaders']
    task_num = len(task_name)

    # define your rl seed
    seed_name = [0, 1]
    seed_num = len(seed_name)

    # produce a schedule
    py_task_schedule = task_schedule(task_name, seed_name)
    # total task num
    py_task_num = len(py_task_schedule)
    # next task idx
    task_idx = 0

    # your python file name
    py_name = '/home/qingyuanwu/wsl-code/MinAtar_DQNs/Implementation/Retrace_Agent.py --game {} --seed {}'

    # create one py controller
    py_controller = py_process_pool(pool_size=1)

    while (1):
        # add one task into py controller
        if (task_idx < py_task_num and py_controller.tasks_num < py_controller.pool_size):
            task_name = py_name.format(py_task_schedule[task_idx][0], py_task_schedule[task_idx][1])
            py_controller.add_process(name=task_name)
            task_idx += 1
        py_controller.remove_dead_process()

        if (task_idx == py_task_num and py_controller.tasks_num == 0):
            break