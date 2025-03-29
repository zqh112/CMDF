import os
from PIL import Image
import augmentations
import numpy as np
from torchvision import transforms

def augment_and_mix(image, preprocess, severity=1, width=3, depth=-1, alpha=1.):
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(preprocess(image))
    for i in range(width):
        image_aug = image.copy()
        d = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(d):
            op = np.random.choice(augmentations.augmentations)
            image_aug = op(image_aug, severity)

            mix = np.clip(mix + ws[i] * preprocess(image_aug).numpy(), 0, 1)

        mixed = np.clip((1 - m) * preprocess(image).numpy() + m * mix, 0, 1)
    mixed_uint8 = (mixed * 255).astype(np.uint8)
    mixed_pil = transforms.ToPILImage()(mixed_uint8.transpose(1, 2, 0))
    return mixed_pil

images_path="/data/name/code/DeepSense6G_TII-main/Dataset/Adaptation_dataset_multi_modal/scenario33/unit1/camera_data/"
images_augmix_path="/data/name/code/DeepSense6G_TII-main/Dataset/Adaptation_dataset_multi_modal/scenario33/unit1/camera_data_augmix/"

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

image_list=os.listdir(images_path)
for image_item in image_list:
    img=images_path+image_item
    img_sample=Image.open(img)

    train_transform = transforms.Compose([
        transforms.CenterCrop(540),
        transforms.RandomHorizontalFlip()
    ])
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    img_sample = train_transform(img_sample)
    img_augmix = augment_and_mix(img_sample, preprocess, severity=1, width=3, depth=-1, alpha=1.)
    img_augmix_path = images_augmix_path+image_item[:-4]+".jpg"
    img_augmix.save(img_augmix_path, 'JPEG')