import re
import matplotlib.pyplot as plt

def plot_from_log(log_path: str):
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []

    pattern = re.compile(
        r"Train Acc: ([0-9.]+) \| Val Acc: ([0-9.]+) \| Train Loss: ([0-9.]+) \| Val Loss: ([0-9.]+)"
    )

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                ta, va, tl, vl = map(float, match.groups())
                train_acc.append(ta)
                val_acc.append(va)
                train_loss.append(tl)
                val_loss.append(vl)

    epochs = range(1, len(train_acc) + 1)

    # 1️⃣ Training Accuracy
    plt.figure()
    plt.plot(epochs, train_acc)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.savefig('train_acc.png')
    plt.close()

    # 2️⃣ Validation Accuracy
    plt.figure()
    plt.plot(epochs, val_acc)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.savefig('val_acc.png')
    plt.close()

    # 3️⃣ Training Loss
    plt.figure()
    plt.plot(epochs, train_loss)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig('train_loss.png')
    plt.close()

    # 4️⃣ Validation Loss
    plt.figure()
    plt.plot(epochs, val_loss)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig('val_loss.png')
    plt.close()


# 🔥 使用方式
plot_from_log('/home/bigrem/CV/hw2_runpod/p2/experiment/mynet_2026_04_02_12_01_31_mynet_custom_resnet9/log/result_log.txt')