"""
Logging utilities - placeholder for Person 2
"""

class Logger:
    """Placeholder logger for testing"""
    def __init__(self, run_dir, config):
        self.run_dir = run_dir
        print(f"[Logger] Initialized at {run_dir}")
    
    def log_train(self, step, metrics):
        print(f"[Train {step}] {metrics}")
    
    def log_eval(self, step, metrics):
        print(f"[Eval {step}] {metrics}")

__all__ = ['Logger']