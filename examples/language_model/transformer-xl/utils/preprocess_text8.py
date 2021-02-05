import sys
import zipfile
import argparse

if __name__ == "__main__":
    data = zipfile.ZipFile("text8.zip").extractall()
    data = open("text8", "r", encoding="utf-8").read()

    num_test_char = int(sys.argv[1])

    train_data = data[:-2 * num_test_char]
    valid_data = data[-2 * num_test_char:-num_test_char]
    test_data = data[-num_test_char:]

    for files, data in [("train.txt", train_data), ("valid.txt", valid_data),
                        ("test.txt", test_data)]:
        data_str = " ".join(["_" if c == " " else c for c in data.strip()])
        with open(files, "w") as f:
            f.write(data_str)
        with open(files + ".raw", "w", encoding="utf-8") as fw:
            fw.write(data)
