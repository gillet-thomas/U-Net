import os
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from CarvanaDataset import CarvanaDataset

import cv2


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # maintains spatial dim to refine features
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)     # [1, 128, 128, 128] -> [1, 256, 128, 128]
        p = self.pool(down)     # [1, 256, 128, 128] -> [1, 256, 64, 64]

        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2): # x1 is the upsampled feature map, x2 is the downsampled feature map (skip connection)
        x1 = self.up(x1)                # [1, 256, 128, 128] -> [1, 128, 256, 256]
        x = torch.cat((x1, x2), dim=1)  # [1, 128, 256, 256] + [1, 128, 256, 256] -> [1, 256, 256, 256]
        x = self.conv(x)                # [1, 256, 256, 256] -> [1, 128, 256, 256]
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)             
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)         # (1, 3, 512, 512) -> (1, 64, 256, 256)
        down_2, p2 = self.down_convolution_2(p1)        # (1, 64, 256, 256) -> (1, 128, 128, 128)
        down_3, p3 = self.down_convolution_3(p2)        # (1, 128, 128, 128) -> (1, 256, 64, 64)
        down_4, p4 = self.down_convolution_4(p3)        # (1, 256, 64, 64) -> (1, 512, 32, 32)

        b = self.bottle_neck(p4)                        # (1, 512, 32, 32) -> (1, 1024, 32, 32)
        
        up_1 = self.up_convolution_1(b, down_4)         # (1, 1024, 32, 32) -> (1, 512, 64, 64)
        up_2 = self.up_convolution_2(up_1, down_3)      # (1, 512, 64, 64) -> (1, 256, 128, 128)
        up_3 = self.up_convolution_3(up_2, down_2)      # (1, 256, 128, 128) -> (1, 128, 256, 256)
        up_4 = self.up_convolution_4(up_3, down_1)      # (1, 128, 256, 256) -> (1, 64, 512, 512)
        
        out = self.out(up_4)                            # (1, 64, 512, 512) -> (1, num_classes, 512, 512)
        return out


def dice_coefficient(prediction, target, epsilon=1e-07):

    prediction_copy = prediction.clone()

    prediction_copy[prediction_copy < 0] = 0
    prediction_copy[prediction_copy > 0] = 1

    intersection = abs(torch.sum(prediction_copy * target))
    union = abs(torch.sum(prediction_copy) + torch.sum(target))
    dice = (2. * intersection + epsilon) / (union + epsilon) # epsilon is added to avoid division by zero
    
    return dice

def train_model(model, train_dataloader, criterion, optimizer, device, epoch):
    model.train()
    train_running_loss = 0
    train_running_dc = 0
    
    # Training
    for idx, img_mask in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        img = img_mask[0].float().to(device)
        mask = img_mask[1].float().to(device)
        
        y_pred = model(img)
        optimizer.zero_grad()
        
        dc = dice_coefficient(y_pred, mask) # Dice coefficient used as performance metric
        loss = criterion(y_pred, mask)
    
        train_running_loss += loss.item()
        train_running_dc += dc.item()

        loss.backward()
        optimizer.step()

        # Visualize mask feature iterations
        if idx == 0:
            pred_mask = y_pred.detach().cpu().permute(0, 2, 3, 1)[0].numpy()           # Take first prediction of first batch
            pred_mask[pred_mask < 0] = 0
            pred_mask[pred_mask > 0] = 1
            cv2.imwrite(f"iterations/training_epoch_{epoch}_pred_mask.png", pred_mask*255)

    train_loss = train_running_loss / (idx + 1)
    train_dc = train_running_dc / (idx + 1)
    
    return train_loss, train_dc

def validate_model(model, val_dataloader, criterion, device):
    model.eval()
    val_running_loss = 0
    val_running_dc = 0
    
    # Validation
    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(val_dataloader, position=0, leave=True)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            loss = criterion(y_pred, mask)
            dc = dice_coefficient(y_pred, mask)
            
            val_running_loss += loss.item()
            val_running_dc += dc.item()

        val_loss = val_running_loss / (idx + 1)
        val_dc = val_running_dc / (idx + 1)
    
    return val_loss, val_dc

def test_model():
    model_pth = '/kaggle/working/my_checkpoint.pth'
    trained_model = UNet(in_channels=3, num_classes=1).to(device)
    trained_model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    test_running_loss = 0
    test_running_dc = 0

    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(test_dataloader, position=0, leave=True)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = trained_model(img)
            loss = criterion(y_pred, mask)
            dc = dice_coefficient(y_pred, mask)

            test_running_loss += loss.item()
            test_running_dc += dc.item()

        test_loss = test_running_loss / (idx + 1)
        test_dc = test_running_dc / (idx + 1)

def random_images_inference(test_dataloader, model_pth, device, n=10):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.Resize((512, 512))
    ])

    # Select random images from the test dataset
    image_tensors = []
    mask_tensors = []
    image_paths = []

    # Randomly select n samples from the test dataset
    for _ in range(n):
        random_index = random.randint(0, len(test_dataloader.dataset) - 1)
        random_sample = test_dataloader.dataset[random_index]

        image_tensors.append(random_sample[0])  
        mask_tensors.append(random_sample[1]) 
        image_paths.append(random_sample[2]) 

    # Create a single figure with subplots for all images
    fig, axes = plt.subplots(n, 3, figsize=(15, 5*n))
    fig.suptitle('Segmentation Results', fontsize=16)
    
    # Iterate for the images, masks and paths
    for i, (image_pth, mask_pth, image_path) in enumerate(zip(image_tensors, mask_tensors, image_paths)):
        # Load the image
        img = transform(image_pth)
        
        # Predict the imagen with the model
        img = img.to(device)  # Move input to the same device as the model
        pred_mask = model(img.unsqueeze(0))
        pred_mask = pred_mask.squeeze(0).permute(1,2,0) # Put channel last [1, 512, 512] -> [512, 512, 1]
        
        # Load the mask to compare
        mask = transform(mask_pth).permute(1, 2, 0).to(device)
        
        # Calculate Dice coefficient
        dice = round(float(dice_coefficient(pred_mask, mask)), 5)
        print(f"Image: {os.path.basename(image_path)}, DICE coefficient: {dice}")
        
        # Prepare images for visualization
        img = img.detach().cpu().permute(1, 2, 0) # Put channel last [3, 512, 512] -> [512, 512, 3]
        pred_mask = pred_mask.detach().cpu()
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 0] = 1
        mask = mask.detach().cpu()
        
        # Plot in the subplot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original: {os.path.basename(image_path)}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(pred_mask, cmap="gray")
        axes[i, 1].set_title(f"Predicted (DICE: {dice})")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(mask, cmap="gray")
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig("results_inference.png")
    plt.show()

def result_plotting(train_losses, val_losses, train_dcs, val_dcs, epochs):
    epochs_list = list(range(1, epochs + 1))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_losses, label='Training Loss')
    plt.plot(epochs_list, val_losses, label='Validation Loss')
    plt.xticks(ticks=list(range(1, epochs + 1, 1))) 
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.tight_layout()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, train_dcs, label='Training DICE')
    plt.plot(epochs_list, val_dcs, label='Validation DICE')
    plt.xticks(ticks=list(range(1, epochs + 1, 1)))  
    plt.title('DICE Coefficient over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('DICE')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('results.png')

if __name__ == "__main__":
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(42)
    num_workers = 4
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-4
    EPOCHS = 10
    MODEL_PATH = 'checkpoint.pth'

    # Load data
    dataset = CarvanaDataset(root_path="/mnt/data/iai/Projects/ABCDE/fmris/CLIP_fmris/segmentation/carvana", limit=100)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=generator)      # 80% training
    test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5], generator=generator)   # 10% validation, 10% testing
    train_dataloader = DataLoader(dataset=train_dataset, num_workers=num_workers, pin_memory=False, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, num_workers=num_workers, pin_memory=False, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, num_workers=num_workers, pin_memory=False, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, optimizer and loss function
    model = UNet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Example usage
    train_losses = []
    train_dcs = []
    val_losses = []
    val_dcs = []

    # Training loop
    for epoch in tqdm(range(EPOCHS)):
        
        # Train the model
        train_loss, train_dc = train_model(model, train_dataloader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        train_dcs.append(train_dc)

        # Validate the model
        val_loss, val_dc = validate_model(model, val_dataloader, criterion, device)
        val_losses.append(val_loss)
        val_dcs.append(val_dc)

        # Print the results
        print("-" * 30)
        print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Training DICE EPOCH {epoch + 1}: {train_dc:.4f}")
        print("\n")
        print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        print(f"Validation DICE EPOCH {epoch + 1}: {val_dc:.4f}")
        print("-" * 30)

        # Saving the model
        torch.save(model.state_dict(), MODEL_PATH)

    # Plot the results
    result_plotting(train_losses, val_losses, train_dcs, val_dcs, EPOCHS)

    # Perform inference on random images
    random_images_inference(test_dataloader, MODEL_PATH, device)