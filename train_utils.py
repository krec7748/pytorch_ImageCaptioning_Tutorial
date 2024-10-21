import numpy as np
import torch
import math
from tqdm import tqdm
# import matplotlib.pyplot as plt

def captioner_train(model, train_DL, val_DL, criterion,
          optimizer = None, scheduler = None, epochs = 10,
          save_model_path = "./results/captioner.pt",
          save_history_path = "./results/captioner_history.pt",
          device = "cpu"):
    
    loss_history = {"train": [], "val": []}
    best_loss = np.inf
    
    for epoch in range(epochs):
        model.train()
        train_loss = loss_epoch(model, train_DL, criterion, optimizer = optimizer, scheduler = scheduler, device = device)
        loss_history["train"] += [train_loss]

        model.eval()
        with torch.no_grad():
            val_loss = loss_epoch(model, val_DL, criterion, device = device)
            loss_history["val"] += [val_loss]
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({"model" : model,
                            "epoch" : epoch,
                            "optimizer" : optimizer,
                            "scheduler" : scheduler},
                            save_model_path)
        
        print(f"Epoch {epoch + 1}: train_loss: {train_loss:.5f}     val_loss: {val_loss:.5f}    current LR: {optimizer.param_groups[0]['lr']:.8f}")
        print("-" * 20)
    
    torch.save({"loss_history" : loss_history,
                "Epochs": epochs}, save_history_path)

def captioner_test(model, test_DL, criterion, device = "cpu"):
    model.eval()
    with torch.no_grad():
        test_loss = loss_epoch(model, test_DL, criterion, device = device)
    print(f"Test loss: {test_loss:.5f} | Test PPL: {math.exp(test_loss):.3f}")


def loss_epoch(model, DL, criterion, optimizer = None, scheduler = None, device = "cpu"):
    N = len(DL.dataset)
    print(device)

    rloss = 0
    for (imgs, src_tokens), trg_tokens in tqdm(DL, leave = False):
        imgs = imgs.to(device)
        src_tokens = src_tokens.to(device)

        # Inference
        y_hat = model(imgs, src_tokens)
        y_hat = y_hat.to("cpu")

        # Loss
        loss = criterion(y_hat.permute(0, 2, 1), trg_tokens)

        # Update
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Loss accumulation
        loss_b = loss.item() * imgs.shape[0]
        rloss += loss_b

    loss_e = rloss / N
    return loss_e

class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps, LR_scale = 1):
        self.optimizer = optimizer
        self.current_step = 0
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.LR_scale = LR_scale
    
    def step(self):
        self.current_step += 1
        lrate = self.LR_scale * (self.d_model ** - 0.5) * min(self.current_step ** -0.5, self.current_step * self.warmup_steps ** -1.5)
        self.optimizer.param_groups[0]["lr"] = lrate

# def plot_scheduler(scheduler_name, optimizer, scheduler, total_steps):
#     lr_history = []
#     steps = range(1, total_steps)

#     for _ in steps:
#         lr_history += [optimizer.param_groups[0]['lr']]
#         scheduler.step()

#     plt.figure()
#     if scheduler_name == 'Noam':
#         if total_steps == 100000:
#             plt.plot(steps, (512 ** -0.5) * torch.tensor(steps) ** -0.5, 'g--', linewidth=1, label=r"$d_{\mathrm{model}}^{-0.5} \cdot \mathrm{step}^{-0.5}$")
#             plt.plot(steps, (512 ** -0.5) * torch.tensor(steps) * 4000 ** -1.5, 'r--', linewidth=1, label=r"$d_{\mathrm{model}}^{-0.5} \cdot \mathrm{step} \cdot \mathrm{warmup\_steps}^{-1.5}$")
#         plt.plot(steps, lr_history, 'b', linewidth=2, alpha=0.8, label="Learning Rate")
#     elif scheduler_name == 'Cos':
#         plt.plot(steps, lr_history, 'b', linewidth=2, alpha=0.8, label="Learning Rate")
#     plt.ylim([-0.1*max(lr_history), 1.2*max(lr_history)])
#     plt.xlabel('Step')
#     plt.ylabel('Learning Rate')
#     plt.grid()
#     plt.legend()
#     plt.show()