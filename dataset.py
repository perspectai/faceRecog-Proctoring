import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import mediapipe as mp
import os
import numpy as np
from torchvision import transforms
from torchvision.utils import make_grid
from itertools import combinations, permutations
from pathlib import Path
import matplotlib.pyplot as plt
import random

class FaceRecognitionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the person subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.persons = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.images_mapping = self._load_images()

    def _load_images(self):
        images_mapping = {}
        for person in self.persons:
            person_dir = Path(self.root_dir) / person
            images = list(person_dir.glob('*'))
            images_mapping[person] = images
        return images_mapping

    def _crop_face(self, image_path):
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            image = cv2.imread(str(image_path))
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.detections:
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                            int(bboxC.width * iw), int(bboxC.height * ih)

                # Adjusting to ensure the crop is within the image
                x, y = max(x, 0), max(y, 0)
                w, h = min(w, iw - x), min(h, ih - y)

                # Proceed only if dimensions are positive
                if w > 0 and h > 0:
                    cropped_image = image[y:y+h, x:x+w]
                    return cropped_image
                else:
                    return None
            return None

    def __len__(self):
        # This could be adjusted based on how you want to sample
        return 2 * len(self.persons)  # Assuming equal number of positive and negative pairs

    def __getitem__(self, idx):
        # Select anchor and positive images
        anchor_person = random.choice(self.persons)
        anchor_images = self.images_mapping[anchor_person]
        while len(anchor_images) < 2:
            # Ensure selected person has at least two images for anchor and positive
            anchor_person = random.choice(self.persons)
            anchor_images = self.images_mapping[anchor_person]
        
        anchor_img_path, positive_img_path = random.sample(anchor_images, 2)
        
        # Select a negative image from a different person
        negative_persons = [p for p in self.persons if p != anchor_person]
        negative_person = random.choice(negative_persons)
        negative_img_path = random.choice(self.images_mapping[negative_person])

        anchor_img = self._crop_face(anchor_img_path)
        positive_img = self._crop_face(positive_img_path)
        negative_img = self._crop_face(negative_img_path)

        # Ensure images are not None
        if anchor_img is None or positive_img is None or negative_img is None:
            # Skip to next item if face detection fails
            return self.__getitem__(idx)

        # Apply transformations
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return (anchor_img, positive_img), (anchor_img, negative_img)




if __name__ == '__main__':

    # Creating some helper functions
    def imshow(img, text=None):
        npimg = img.numpy()
        plt.axis("off")
        if text:
            plt.text(75, 8, text, style='italic',fontweight='bold',
                bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        plt.imsave('test.png', np.transpose(npimg, (1, 2, 0)))  


    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((160, 160), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    dataset = FaceRecognitionDataset(root_dir='train_ds', transform=transform)

    vis_dataloader = DataLoader(dataset,
                        shuffle=True,
                        num_workers=2,
                        batch_size=8)
    

    # Extract one batch
    example_batch = next(iter(vis_dataloader))

    concatenated = torch.cat((example_batch[0], example_batch[1]),0)

    imshow(make_grid(concatenated))
    print(example_batch[2].numpy().reshape(-1))


