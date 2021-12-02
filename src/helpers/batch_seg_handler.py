class BatchHandler:
    def __init__(self, mode, mode_args, max_len=None):
        self.device = torch.device('cpu')
        self.mode = mode
        self.max_len = max_len
    
    