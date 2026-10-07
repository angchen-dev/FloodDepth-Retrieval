import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models import vgg19, VGG19_Weights
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg19
from PIL import Image
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models import vgg19, VGG19_Weights


def load_mask_rcnn(device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    maskrcnn_weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1  # 或 .DEFAULT
    maskrcnn_model = maskrcnn_resnet50_fpn(weights=maskrcnn_weights).to(device)
    maskrcnn_model.eval();
    return maskrcnn_model


def load_image(image_path):
    """Load an image from the path and convert it to a tensor."""
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size  # Get original dimensions
        
        # Resize if width is greater than 960 while maintaining aspect ratio
        if width > 960:
            new_width = 960
            new_height = int((new_width / width) * height)  # Preserve aspect ratio
            img = img.resize((new_width, new_height), Image.LANCZOS)

        if height > 540:
            new_height = 540
            new_width = int((new_height / height) * width)  # Preserve aspect ratio
            img = img.resize((new_width, new_height), Image.LANCZOS)

        img_tensor = F.to_tensor(img)
        return img, img_tensor
    except IOError as e:
        print(f"Error opening image: {e}")
        return None, None

def apply_maskrcnn(model, image_tensor):
    """Apply the Mask R-CNN model to the tensorized image."""
    if image_tensor is None:
        print("Invalid image tensor, skipping...")
        return None
    model.eval()  # Set the model to inference mode
    with torch.no_grad():
        device = next(model.parameters()).device
        predictions = model([image_tensor.to(device)])  # Assume the model and tensor are on the same device

        predictions = predictions[0]

    return predictions


def filter_occlusion(img_pil, predictions, mask_threshold=0.2, score_thresh=0.9):
    """
    Returns:
        - Filtered car candidates (each with bbox and mask)
        - Visualization image with all object bboxes in red and all car bboxes in yellow
        - Visualization image with only filtered car bboxes in yellow
        - Occlusion count
    All detected objects are drawn in red, all cars are drawn in yellow (overlapping).
    """
    from PIL import ImageDraw

    cnt = 0

    # Confidence filtering
    scores = predictions['scores'].cpu().numpy()
    keep = scores > score_thresh

    boxes_all = predictions['boxes'][keep].cpu().numpy().astype(int)
    labels_all = predictions['labels'][keep].cpu().numpy()
    masks_all = (predictions['masks'][keep].squeeze(1) > mask_threshold).cpu().numpy()

    # Get all indices where label==3 (car class)
    car_indices = [i for i, l in enumerate(labels_all) if l == 3]
    if not car_indices:
        return []

    candidate_list = set(car_indices)

    for idx in car_indices:
        xi1, yi1, xi2, yi2 = boxes_all[idx]
        yb_i = yi2
        for j, (xj1, yj1, xj2, yj2) in enumerate(boxes_all):
            if j == idx:
                continue
            is_overlap = (xi1 < xj2 and xi2 > xj1 and yi1 < yj2 and yi2 > yj1)
            if is_overlap:
                yb_j = yj2
                if yb_i < yb_j:
                    candidate_list.discard(idx)
                    cnt += 1
                    break

    # Build result: a list of candidate cars (bbox, mask)
    result = []
    for idx in sorted(candidate_list):
        result.append({
            'bbox': tuple(boxes_all[idx]),
            'mask': masks_all[idx]
        })

    # Draw all detected objects in red, all cars in yellow (on top)
    draw_img_all_cars = img_pil.copy()
    draw_all = ImageDraw.Draw(draw_img_all_cars)
    # All objects (any label) in red
    for idx in range(len(boxes_all)):
        x1, y1, x2, y2 = boxes_all[idx]
        draw_all.rectangle([x1, y1, x2, y2], outline='red', width=3)
    # All cars (label==3) in yellow
    for idx in car_indices:
        x1, y1, x2, y2 = boxes_all[idx]
        draw_all.rectangle([x1, y1, x2, y2], outline='yellow', width=3)

    # Draw candidate car boxes in yellow (after filtering)
    draw_img_candidates = img_pil.copy()
    draw_cand = ImageDraw.Draw(draw_img_candidates)
    for car in result:
        x1, y1, x2, y2 = car['bbox']
        draw_cand.rectangle([x1, y1, x2, y2], outline='yellow', width=4)

    return result, draw_img_all_cars, draw_img_candidates, cnt


def filter_candidates_on_edge(candidates, img_pil, margin=0):
    """
    Remove candidates whose mask touches the edge of the image.
    Args:
        candidates: list of {'bbox': ..., 'mask': ...}
        img_w_h: (width, height)
        margin: integer, margin to define 'edge'
        img_pil: (optional) PIL image for visualization
    Returns:
        filtered: candidate list after edge removal
        draw_img: PIL image with red (all) and yellow (final) boxes
    """
    from PIL import ImageDraw

    w, h = img_pil.size
    # List for candidates that are not on the edge
    filtered = []
    for car in candidates:
        mask = car['mask']
        pos = np.where(mask)
        if len(pos[0]) == 0 or len(pos[1]) == 0:
            continue
        y_min, y_max = pos[0].min(), pos[0].max()
        x_min, x_max = pos[1].min(), pos[1].max()
        if (x_min <= margin or x_max >= w - margin or
            y_min <= margin or y_max >= h - margin):
            continue  # Skip: is on edge
        filtered.append(car)

    # Visualization: red for original, yellow for filtered
    draw_img = img_pil.copy() if img_pil else None
    if draw_img:
        draw = ImageDraw.Draw(draw_img)
        for car in filtered:
            x1, y1, x2, y2 = car['bbox']
            draw.rectangle([x1, y1, x2, y2], outline='yellow', width=4)

    return filtered, draw_img

def filter_candidates_by_wh_ratio(candidates, img_pil, ratio_thresh=1.28):
    """
    Remove candidates whose mask's min bbox width/height ratio is less than ratio_thresh.
    Also returns a visualization image (red: all input, yellow: after filtering).
    Args:
        candidates: list of {'bbox': ..., 'mask': ...}
        img_pil: PIL.Image, for drawing
        ratio_thresh: float
    Returns:
        filtered: candidate list after filtering by ratio
        draw_img: PIL.Image with red (all) and yellow (filtered) boxes
    """
    from PIL import ImageDraw

    filtered = []
    for car in candidates:
        mask = car['mask']
        pos = np.where(mask)
        if len(pos[0]) == 0 or len(pos[1]) == 0:
            continue
        h = pos[0].max() - pos[0].min() + 1
        w = pos[1].max() - pos[1].min() + 1
        ratio = w / h
        if ratio >= ratio_thresh:
            filtered.append(car)

    # Visualization: red for input, yellow for filtered
    draw_img = img_pil.copy()
    draw = ImageDraw.Draw(draw_img)
    # Draw all original candidates (input) in red
    for car in filtered:
        x1, y1, x2, y2 = car['bbox']
        draw.rectangle([x1, y1, x2, y2], outline='yellow', width=4)

    return filtered, draw_img


def select_largest_vehicle(candidates, img_pil):
    max_area = -1

    final_candidate = []
    largest_car = None
    for car in candidates:
        mask = car['mask']
        if mask is not None:
            area = np.sum(mask)
            if area > max_area:
                max_area = area
                largest_car = car
    
    if largest_car is not None:
        final_candidate.append(largest_car )

    # Visualization: red for input, yellow for filtered
    draw_img = img_pil.copy()
    draw = ImageDraw.Draw(draw_img)
    # Draw all original candidates (input) in red
    for car in final_candidate:
        x1, y1, x2, y2 = car['bbox']
        draw.rectangle([x1, y1, x2, y2], outline='yellow', width=4)

    return final_candidate, draw_img


def save_draw_bbox(draw_img_all_cars, draw_img_occ, draw_img_edge, draw_img_ratio, draw_img_final, save_path):
    plt.figure(figsize=(30, 10))

    plt.subplot(1, 5, 1)
    plt.imshow(draw_img_all_cars)
    plt.title("Detected cars")
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.imshow(draw_img_occ)
    plt.title("After occlusion")
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.imshow(draw_img_edge)
    plt.title("After on edge")
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.imshow(draw_img_ratio)
    plt.title("After ratio")
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(draw_img_final)
    plt.title("Final")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def crop_and_resize_with_margin(img_pil, mask, width=212, target_size=224):
    """
    Crop the vehicle from the image according to the segmentation mask, resize to the given width 
    (maintaining aspect ratio), and pad with white to create a target_size x target_size image.
    Returns: (new_img, new_height, margin), where
        - new_img: the processed Image object,
        - new_height: the resized vehicle height in pixels,
        - margin: actual top (and left/right) margin in pixels.
    """

    # Create white background
    blank_img = Image.new('RGB', img_pil.size, (255, 255, 255))
    
    # Prepare mask as uint8 PIL image
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_uint8, mode='L')
    
    # Composite: vehicle-only image, background set to white
    vehicle_img = Image.composite(img_pil, blank_img, mask_img)
    
    # Find bounding box from mask
    pos = np.where(mask)
    y_min, y_max = pos[0].min(), pos[0].max()
    x_min, x_max = pos[1].min(), pos[1].max()
    
    # Crop to bounding box
    crop = vehicle_img.crop((x_min, y_min, x_max+1, y_max+1))
    
    # Resize (width fixed, height proportional)
    scale = width / (x_max - x_min + 1)
    new_height = int((y_max - y_min + 1) * scale)
    car_img_resized = crop.resize((width, new_height), resample=Image.BILINEAR)
    
    # Compute margin for centering
    margin = (target_size - width) // 2
    
    # Place resized car on white background, top and left margins = margin
    new_img = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    new_img.paste(car_img_resized, (margin, margin))

    new_img_np = np.array(new_img)
    
    return new_img_np, new_height, margin



def save_matched_result(query_img_np, matched_img_np, bottom_text, save_path):
    plt.figure(figsize=(6, 3))

    # 1. First Subplot (left image)
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, this is the 1st plot
    plt.imshow(query_img_np)
    plt.title("Query Image")
    plt.axis('off')  # Hide the axes for a cleaner look

    # 2. Second Subplot (right image)
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, this is the 2nd plot
    plt.imshow(matched_img_np)
    plt.title("Matched Image")
    plt.axis('off')  # Hide the axes
    plt.figtext(0.5, 0.01, bottom_text, ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



class DFSSImageRetrieval:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize DFSS Image Retrieval with VGG19
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = self._load_vgg19_features()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def _load_vgg19_features(self):
        """Load VGG19 and extract features up to pooling layer 4"""
        vgg19_model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).to(self.device)
        vgg19_model.eval();


        # vgg = vgg19(pretrained=True)
        
        # Extract features up to pooling layer 4 (pool4)
        # VGG19 architecture: conv-relu-conv-relu-pool (x5 blocks)
        # Pool4 is after the 4th pooling layer
        features = list(vgg19_model.features.children())
        
        # Find pooling layer 4 (index 27 in VGG19 features)
        # pool1: idx 4, pool2: idx 9, pool3: idx 18, pool4: idx 27
        pool4_idx = 27
        
        model = nn.Sequential(*features[:pool4_idx+1])
        model.eval()
        model.to(self.device)
        
        return model
    
    
    def _extract_features(self, image):
        # Preprocess image for VGG19
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Extract features from pooling layer 4
        with torch.no_grad():
            features = self.model(input_tensor)

        return features

    def _get_dfss_height_ratio(self, image):
        # Ensure image is in correct format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        # Get vehicle content area within margins (skip top=6, left=6, right=6)
        height, width = image.shape[:2]
        top_margin = 6
        left_margin = 6 
        right_margin = 6
        
        # Find actual vehicle content boundary within the margins
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        white_threshold = 240
        min_pixels = max(1, (width - left_margin - right_margin) // 20)
        
        last_content_row = top_margin
        
        # Scan from top_margin downward, only within left/right margins
        for y in range(top_margin, height):
            row_segment = gray[y, left_margin:width-right_margin]
            non_white_count = np.sum(row_segment < white_threshold)
            if non_white_count >= min_pixels:
                last_content_row = y
        
        # Calculate content ratio: from top of image to last content row
        content_height_ratio = (last_content_row + 1) / height


        return top_margin, left_margin, right_margin, content_height_ratio

    
    def _extract_dfss_features(self, features,top_margin, left_margin, right_margin, content_height_ratio):
        """
        Extract DFSS features from pooling layer 4, focusing on content within margins
        All images have left=6, right=6, top=6 margins, vehicle content is within these margins
        
        Args:
            image: Input image as numpy array (H, W, C) or (H, W)
            
        Returns:
            numpy.ndarray: Flattened feature vector
        """
        
        # Apply DFSS: use content area ratio to crop feature map
        # Pooling layer 4 output is 1/16 of original size (14x14 for 224x224 input)
        feature_height, feature_width = features.shape[2], features.shape[3]
        
        # Calculate feature map crop dimensions based on margins
        # For 224x224 -> 14x14 feature map: margin of 6 pixels -> ~0.4 feature pixels
        top_feature_margin = max(0, int(top_margin / 14))  # 6/16 ≈ 0.4 -> 0
        left_feature_margin = max(0, int(left_margin / 14))  # 6/16 ≈ 0.4 -> 0  
        right_feature_margin = max(0, int(right_margin / 14))  # 6/16 ≈ 0.4 -> 0
        
        # Height crop based on content ratio
        content_feature_height = max(1, int(feature_height * content_height_ratio))
        
        # Width crop based on margins (though margins are small, keep most width)
        content_feature_width = feature_width - left_feature_margin - right_feature_margin
        content_feature_width = max(1, content_feature_width)
        
        # Extract DFSS features: content area only
        dfss_features = features[:, :, 
                               top_feature_margin:content_feature_height,
                               left_feature_margin:left_feature_margin + content_feature_width]
        
        # Global Average Pooling to get fixed-size feature vector
        pooled_features = torch.mean(dfss_features, dim=(2, 3))

        return pooled_features.cpu().numpy().flatten()
    
    def image_retrieval(self, query_image_np:np.array, training_images_np:list, training_features:list, top_k=5):
        """
        Perform image retrieval using DFSS method
        
        Args:
            query_image: Query image as numpy array
            training_images: List of training images as numpy arrays
            top_k: Number of top similar images to return
            
        Returns:
            list: Indices of most similar training images, sorted by similarity
        """
        # print(f"Extracting query features...")
        query_features = self._extract_features(query_image_np)

        top_margin, left_margin, right_margin, query_height_ratio = self._get_dfss_height_ratio(query_image_np)

        query_features_dfss = self._extract_dfss_features(query_features, top_margin, left_margin, right_margin, query_height_ratio)
        query_features_dfss = query_features_dfss.reshape(1, -1)

        training_f_dfss_all = []
        train_height_ratio_all = []

        for training_image_np, training_f in zip(training_images_np, training_features):
            training_f_dfss_all.append(
                self._extract_dfss_features(
                    training_f, top_margin, left_margin, right_margin, query_height_ratio)
                    )

            _, _, _, train_height_ratio = self._get_dfss_height_ratio(training_image_np)
            train_height_ratio_all.append(train_height_ratio)

        training_f_dfss_all = np.array(training_f_dfss_all)


        # Calculate cosine similarity
        query_features_dfss = query_features_dfss.reshape(1, -1)
        similarities = cosine_similarity(query_features_dfss, training_f_dfss_all).flatten()
        
        # Get top-k most similar images
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]

        top_indices = top_indices.tolist()
        top_similarities = top_similarities.tolist()

        diff_pixel = None

        if top_similarities[0] < 0.55:
            top_indices = []
            top_similarities = []
        else:
            top_indices = [top_indices[0]]
            top_similarities = [top_similarities[0]]

            train_height_ratio = train_height_ratio_all[top_indices[0]]

            h,w,_ = training_images_np[top_indices[0]].shape


            diff_pixel = (train_height_ratio - query_height_ratio) * h


        return top_indices, top_similarities, diff_pixel

