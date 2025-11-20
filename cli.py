# cli.py
import os
import subprocess

def run_cmd(cmd):
    print(f"\n>>> Running: {cmd}\n")
    subprocess.run(cmd, shell=True)

def main():
    while True:
        print("\n===== Multi-Agent RL Launcher =====")
        print("1. Train Simple Tag")
        print("2. Evaluate Simple Tag")
        print("3. Train Pistonball")
        print("4. Evaluate Pistonball")
        print("5. Exit")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            run_cmd("python -m training.train_shared --config configs/simple_tag.yaml")

        elif choice == "2":
            run_cmd("python -m eval.evaluate --config configs/simple_tag.yaml --model runs/simple_tag_shared/final_model.zip")

        elif choice == "3":
            run_cmd("python -m training.train_shared --config configs/pistonball.yaml")

        elif choice == "4":
            run_cmd("python -m eval.evaluate --config configs/pistonball.yaml --model runs/pistonball_shared/final_model.zip")

        elif choice == "5":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
