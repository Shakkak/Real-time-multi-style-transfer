from torchvision import transforms

def custom_transform(image_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    composed_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return composed_transform()