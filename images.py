import os
import cv2
import mediapipe as mp

ALL_IMAGES = "train_ds"

min_imgs_each_person = 100000000
max_imgs_each_person = 0
total_images = 0
widths = []
heights = []

mp_face_detection = mp.solutions.face_detection


def _crop_face(image_path):
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


for img_folder in os.listdir(ALL_IMAGES):

    images_path = os.path.join(ALL_IMAGES, img_folder)
    num_images = len(os.listdir(images_path))
    max_imgs_each_person = max(max_imgs_each_person, num_images)
    min_imgs_each_person = min(min_imgs_each_person, num_images)
    total_images += num_images # 2 * ((num_images * (num_images - 1))/2) * int(num_images > 1)
    for image in os.listdir(images_path):
        cropped_face = _crop_face(os.path.join(images_path, image))
        if cropped_face is not None:
            widths.append(cropped_face.shape[1])
            heights.append(cropped_face.shape[0])
            # if cropped_face.shape[0] > 256 and cropped_face.shape[1] > 256:
            #     print(os.path.join(images_path, image) , cropped_face.shape)


                          
    
print('Total Number of people: ', len(os.listdir(ALL_IMAGES)))
print('Total Number of images: ', total_images)
print(f"Maximum width size: {max(widths)} \nMinimum width size: {min(widths)}")
print(f"Maximum height size: {max(heights)} \nMinimum height size: {min(widths)}")
print(f"Average width size: {sum(widths) // len(widths)} \n Average height size: {sum(heights) // len(heights)}")
print('Maximum number of images of each person: ', max_imgs_each_person)
print('Minimum number of images of each person: ', min_imgs_each_person)