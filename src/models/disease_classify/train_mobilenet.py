import argparse
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    from common import train

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["v1", "v2"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input", choices=["square", "wide", "wide_hr"], default="square")
    args = parser.parse_args()
    train("mobilenet_v2", args.data, seed=args.seed, input_mode=args.input)


if __name__ == "__main__":
    main()