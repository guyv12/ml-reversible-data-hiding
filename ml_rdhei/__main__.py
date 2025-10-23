from data.test import test_data_ref, test_data_raw


def main() -> None:

    for i in range(10):
        test_data_ref()

    for i in range(10):   
        test_data_raw()


if __name__ == "__main__":
    main()
