import run
import data
import re
import timeit


def main() -> None:
    run.set_device()
    
    data.set_mask()
    BOSSbase_train_dataset = data.ImageDataset("datasets/BOSSbase_512", re.compile(r"[0-4][0-9]?[0-9]?[0-9]?\.pgm"))
    loader_ref = data.get_loader_ref(BOSSbase_train_dataset)
    loader_raw = data.get_loader_raw(BOSSbase_train_dataset)

    print(f"time ref: {timeit.timeit(lambda: data.test_ref(loader_ref, len(BOSSbase_train_dataset)), number=3)}")
    print(f"time raw: {timeit.timeit(lambda: data.test_raw(loader_raw, len(BOSSbase_train_dataset)), number=3)}")

if __name__ == "__main__":
    main()
