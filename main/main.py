import argparse
import torch
from train.train_cnn import train_model
from inference.infer_cnn import infer_cnn


def main():
    parser = argparse.ArgumentParser(description="Rock-Scissor-Paper Recognition")
    parser.add_argument("--mode",
                        choices=["train", "infer"],
                        default="infer",
                        help="실행 모드: train 또는 infer")
    args = parser.parse_args()

    if args.mode == "train":
        print("🔧 CNN 모델 학습 시작...")
        train_model()

    elif args.mode == "infer":
        print("🎮 CNN 모델 추론 시작...")
        infer_cnn()


if __name__ == "__main__":
    main()