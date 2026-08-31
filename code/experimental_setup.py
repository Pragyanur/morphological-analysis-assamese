import torch
import argparse


def count_state_dict_parameters(state_dict):
    total_params = 0

    print("\n========== LAYER-WISE BREAKDOWN ==========\n")

    for name, tensor in state_dict.items():
        num_params = tensor.numel()
        total_params += num_params

        print(
            f"{name:50s} "
            f"Shape: {str(list(tensor.shape)):20s} "
            f"Params: {num_params:,}"
        )

    print("\n========== MODEL PARAMETER REPORT ==========")
    print(f"Total Parameters: {total_params:,}")

    model_size_mb = total_params * 4 / (1024 ** 2)
    print(f"Approx Model Size (FP32): {model_size_mb:.4f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model.pth"
    )

    args = parser.parse_args()

    state_dict = torch.load(args.model, map_location="cpu")

    count_state_dict_parameters(state_dict)


if __name__ == "__main__":
    main()