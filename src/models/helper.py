class lrd:

    def __init__(self, waiting_time, start_lr, min_lr, factor):
        self.waiting_time = waiting_time
        self.start_lr = start_lr
        self.min_lr = min_lr
        self.factor = factor
        self.min_value = 10000000000000000000
        self.waited = 0
        self.actual_lr = self.start_lr
        self.stop = False

    def set_new_lr(self, new_value):
        if new_value < self.min_value:
            self.waited = 0
            self.min_value = new_value
            return self.actual_lr

        self.waited += 1

        if self.waited > self.waiting_time:
            self.actual_lr /= self.factor
            self.waited = 0
            if self.actual_lr < self.min_lr:
                self.stop = True

        return self.actual_lr
