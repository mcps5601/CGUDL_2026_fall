import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def train_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> float:
    """
    執行一個訓練 epoch 的函式
    """
    model.train() # 確保 model 是在訓練模式
    total_loss = 0 # 紀錄 loss 數值

    progress_bar = tqdm(data_loader, desc="Training")
    for batch in progress_bar:
        # 把資料移動到 GPU
        images, labels = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()

        # 1. 前向傳播
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # 2. 反向傳播 (計算梯度)
        loss.backward()

        # 3. 更新模型權重
        optimizer.step()

        # 儲存 loss 數值
        total_loss += loss.item()

        progress_bar.set_postfix({
            'loss': f"{total_loss / (progress_bar.n + 1):.4f}",
        })
    return total_loss / len(data_loader) # 回傳平均每筆資料的 loss


def do_eval(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, list, list]:
    """
    執行一個評估 epoch 的函式
    """
    model.eval() # 確保 model 是在評估模式
    total_loss = 0 # 紀錄 loss 數值
    label_list = []
    prediction_list = []

    progress_bar = tqdm(data_loader, desc="Evaluating")
    with torch.no_grad():
        for batch in progress_bar:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(images) # `outputs` 的形狀是 (batch_size, 10)
            # print(outputs.shape)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            prediction = outputs.argmax(dim=1) # `prediction` 的型態是 tensor

            # 我們希望 `predictions` 的長相是 [1, 2, 0, 4, 5, 7, ...]
            # 如果用 append 的話可能會變成 [[1, 2, 0], [4, 5, 7], ...]
            prediction_list.extend(prediction.tolist())

            label_list.extend(labels.tolist())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(label_list, prediction_list)
    return avg_loss, accuracy, label_list, prediction_list


def plot_loss_history(train_losses: list, val_losses: list, save_name: str = "loss.png") -> None:
    """
    繪製訓練過程的 Training Loss 和 Validation Loss 曲線。

    Args:
        train_losses (list): 訓練集的 loss 歷史記錄。
        val_losses (list): 驗證集的 loss 歷史記錄。
    """
    # 繪製 loss 的歷史記錄圖
    epochs = len(train_losses)
    assert len(train_losses) == len(val_losses)

    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.xlim([1, epochs])
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(save_name, bbox_inches='tight') # 儲存圖片
    plt.show()