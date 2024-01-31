"""entry function"""
from llmtuner import run_exp


def main():
    """main"""
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
