"""Local runtime helpers from the archived three-corpus training implementation."""
import os
import time


class BudgetStop(RuntimeError):
    pass


def atomic_replace(source, destination):
    """Retry transient Windows indexer/antivirus locks while preserving atomic writes."""
    for attempt in range(100):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if os.name != 'nt' or attempt == 99:
                raise
            time.sleep(.05)


def budget_requested():
    deadline = os.environ.get('MDLM_TRAIN_DEADLINE')
    return deadline is not None and time.time() >= float(deadline)


def require_time():
    if budget_requested():
        raise BudgetStop('Training deadline reached. Resume from durable checkpoints with a new budget.')
