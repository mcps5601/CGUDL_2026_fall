import torch
from sklearn.metrics import f1_score
from tqdm import tqdm


def do_test(
        dataloader,
        model,
        model_type,
        loss_fn,
        device,
        num_epochs,
        cur_epoch=0,
        mode="validation",
):
    model.eval()

    pbar = tqdm(dataloader)
    pbar.set_description(f"{mode} epoch [{cur_epoch+1}/{num_epochs}]")

    pred = torch.tensor([], dtype=torch.int64)
    gt = torch.tensor([], dtype=torch.int64)
    total_loss = 0

    with torch.no_grad():
        for pixel_values, labels in pbar:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)  # 複製通道 3 次
            pixel_values = pixel_values.to(device)
            labels = labels.squeeze().to(device)
            # print(pixel_values.shape)
            # print(labels.shape)

            if model_type == "HF":
                outputs = model(pixel_values=pixel_values, labels=labels).logits
            elif model_type == "custom":
                outputs = model(pixel_values=pixel_values)

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=-1)
            pred = torch.cat((pred, preds.cpu()))
            gt = torch.cat((gt, labels.cpu()))

    accuracy = (pred == gt).float().mean().item()
    f1 = f1_score(gt.numpy(), pred.numpy(), average='macro')

    print(f"Accuracy: {accuracy:.4f} \nF1 Score: {f1:.4f}")
    total_loss /= len(dataloader)
    return total_loss