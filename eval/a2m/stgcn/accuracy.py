import torch
from pdb import set_trace as st



def calculate_accuracy(model, motion_loader, num_labels, classifier, device):
    confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)
    with torch.no_grad():
        for batch in motion_loader:
            batch_prob = classifier(batch)["yhat"]             
            batch_pred = batch_prob.max(dim=1).indices
            for label, pred in zip(batch["y"], batch_pred):
                confusion[label][pred] += 1
    accuracy = torch.trace(confusion)/torch.sum(confusion)
    return accuracy.item(), confusion


# batch 
# dict_keys(['output', 'lengths', 'y', 'features', 'yhat'])



def calculate_accuracy_humanEva(model, motion_loader, num_labels, classifier, device):
    confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)
    with torch.no_grad():
        for batch in motion_loader:
            batch_prob = classifier(batch)["yhat"] 
            
            batch_pred = batch_prob.max(dim=1).indices
            for label, pred in zip(batch["y"], batch_pred):
                # only count the 5 labels # drink:2, blue:12,finish:9, walk:17, can:29
                # st()
                checklist= [2, 9, 12, 17, 29]
                if label.item() in checklist:
                    confusion[label][pred] += 1
    
    accuracy = torch.trace(confusion)/torch.sum(confusion)
    return accuracy.item(), confusion