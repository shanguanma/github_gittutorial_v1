#!/usr/bin/env python3

import torch
import logging
import os


def cpu_test():
    for i in range(10000):
        A = torch.randn(1000, 200, 300)
        B = torch.randn(1000, 300, 300)
        C = torch.matmul(A, B)
        print("{0} C is {1}".format(i, C))


def gpu_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(1000):
        A = torch.randn(100, 200, 300, device=device)
        B = torch.randn(100, 300, 300, device=device)
        C = torch.matmul(A, B)
        print("torch version is {}".format(torch.__version__))
        print("GPU test {0} C is {1}".format(i, C))


if __name__ == "__main__":
    # display PYTHONPATH
    logging.info("python path = " + os.environ.get("PYTHONPATH", "(None)"))
    # cpu_test()
    gpu_test()
    logging.info("python path = " + os.environ.get("PYTHONPATH", "(None)"))
