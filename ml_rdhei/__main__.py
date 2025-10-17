import torch
import timeit
import data


def main() -> None:

    dev = 'cuda' if torch.cuda.is_available() else 'cpu' # kinda essential - to elevate elsewhere later?
    print(torch.version.cuda) # sanity check

    train_set_ref = data.get_train_ref(torch.device(dev))
    print(f"shape ref: {train_set_ref.shape}")

    train_set_raw = data.get_train_raw(torch.device(dev))
    print(f"shape raw: {train_set_raw.shape}")

    print(f"time ref: {timeit.timeit(lambda: data.get_train_ref(torch.device(dev)), number=3)}")
    print(f"time raw: {timeit.timeit(lambda: data.get_train_ref(torch.device(dev)), number=3)}")

if __name__ == "__main__":
    main()
