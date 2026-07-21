import torch

def evaluate(model, test_loader, device):
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:

            
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_features)

            predicted = outputs.argmax(dim=1)

            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    return correct / total