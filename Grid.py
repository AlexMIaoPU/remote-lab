import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import torch
from torchvision import models
import torch.nn as nn

from torchvision import transforms
from PIL import Image


# Used to repesent socket location on breadboard
class Point:
    def __init__(self, x, y, row_idx, col_idx):
        self.x = x
        self.y = y
        self.row_idx = row_idx
        self.col_idx = col_idx

    def __repr__(self):
        return f"Point on row {self.row_idx}, col {self.col_idx} of ({self.x}, {self.y})"
    
    def get_coordinates(self):
        return (self.x, self.y)
    
    def get_index(self):
        return (self.row_idx, self.col_idx)


# The GP type correesponds to the index in class_names
# 0: not_plugged, 1: pass_over, 2: plugged
class_names = ['not_plugged', 'pass_over', 'plugged']

class GridPoint:
    def __init__(self, point, type, probs):
        self.point = point  # Point object
        self.type = type
        self.probs = probs  # Probability distribution over classes

    def __repr__(self):
        return f"GridPoint of type {self.type} at {self.point}"
    
    def get_coordinates(self):
        return self.point.get_coordinates()
    
    def get_index(self):
        return self.point.get_index()
    
    def is_plugged(self):
        return self.type == 2  # Assuming 'plugged' is class index 2

# === 1. Extract Keypoint Coordinates ===
def get_keypoint_coords(keypoints):
    return np.array([kp.pt for kp in keypoints])

# === 2. Cluster Rows (along Y) ===
def cluster_rows(pts, eps=3):
    clustering = DBSCAN(eps=eps, min_samples=4).fit(pts[:, 1].reshape(-1, 1))
    return clustering.labels_

# === 3. Cluster Columns (along X) ===
def cluster_columns(pts, eps=3):
    clustering = DBSCAN(eps=eps, min_samples=4).fit(pts[:, 0].reshape(-1, 1))
    return clustering.labels_

# === 4. Fit Row Lines using RANSAC (y = mx + b) ===
def fit_horizontal_lines(pts, labels):
    lines = []
    for label in np.unique(labels):
        if label == -1:
            continue
        row_pts = pts[labels == label]
        X = row_pts[:, 0].reshape(-1, 1)  # x
        y = row_pts[:, 1]                # y

        model = RANSACRegressor().fit(X, y)
        slope = model.estimator_.coef_[0]
        intercept = model.estimator_.intercept_
        lines.append((slope, intercept))
    return lines

# === 5. Fit Column Lines using RANSAC (x = m*y + b) ===
def fit_vertical_lines(pts, labels, min_points=3):
    lines = []
    for label in np.unique(labels):
        if label == -1:
            continue
        col_pts = pts[labels == label]
        if len(col_pts) < min_points:
            continue
        Y = col_pts[:, 1].reshape(-1, 1)  # y
        X = col_pts[:, 0]                # x

        model = RANSACRegressor().fit(Y, X)
        slope = model.estimator_.coef_[0]
        intercept = model.estimator_.intercept_
        lines.append((slope, intercept))
    return lines

# === 6. Generate Line Points ===
def line_points(slope, intercept, x_range):
    x = np.linspace(*x_range, num=2)
    y = slope * x + intercept
    return np.column_stack([x, y])

def vertical_line_points(slope, intercept, y_range):
    y = np.linspace(*y_range, num=2)
    x = slope * y + intercept
    return np.column_stack([x, y])

# === 7. Drawing Functions ===
def draw_lines(image, lines, x_range, color):
    for slope, intercept in lines:
        pts = line_points(slope, intercept, x_range).astype(int)
        cv2.line(image, tuple(pts[0]), tuple(pts[1]), color, 1)

def draw_vertical_lines(image, lines, y_range, color):
    for slope, intercept in lines:
        pts = vertical_line_points(slope, intercept, y_range).astype(int)
        cv2.line(image, tuple(pts[0]), tuple(pts[1]), color, 1)

def find_intersections(row_lines, col_lines):
    intersections = []
    for row_idx, (m_row, b_row) in enumerate(row_lines):
        for col_idx, (m_col, b_col) in enumerate(col_lines):
            # row: y = m_row * x + b_row
            # col: x = m_col * y + b_col
            # Solve for x and y
            # Substitute x from col into row:
            # y = m_row * (m_col * y + b_col) + b_row
            # y - m_row * m_col * y = m_row * b_col + b_row
            # y * (1 - m_row * m_col) = m_row * b_col + b_row
            denom = 1 - m_row * m_col
            if abs(denom) < 1e-6:
                continue  # Lines are nearly parallel, skip
            y = (m_row * b_col + b_row) / denom
            x = m_col * y + b_col
            
            new_point = Point(x, y, row_idx, col_idx)

            intersections.append(new_point)
    return np.array(intersections)

# intersections is an np array of (x,y) pairs
def get_grid_size(row_count, col_count, intersections):
    """
    intersections: np.array of Point objects (length row_count * col_count)
    Returns average x and y grid spacing.
    """
    if len(intersections) < 2 or row_count < 2 or col_count < 2:
        return None, None

    # Convert to array of (row_idx, col_idx, x, y)
    arr = np.array([[pt.row_idx, pt.col_idx, pt.x, pt.y] for pt in intersections])

    # Sort by row_idx, then col_idx
    arr = arr[np.lexsort((arr[:,2], arr[:,3]))]

    # Reshape to (row_count, col_count, 2) for x and y
    grid_x = arr[:,2].reshape((row_count, col_count))
    grid_y = arr[:,3].reshape((row_count, col_count))

    # Compute average x spacing (horizontal, along columns)
    x_diffs = []
    for r in range(row_count):
        for c in range(col_count - 1):
            x1 = grid_x[r, c]
            x2 = grid_x[r, c + 1]
            x_diffs.append(abs(x2 - x1))
    avg_x_diff = np.mean(x_diffs)

    return avg_x_diff

def print_column_clusters(pts, col_labels):
    for label in np.unique(col_labels):
        if label == -1:
            continue
        col_pts = pts[col_labels == label]
        print(f"Column cluster {label}:")
        for pt in col_pts:
            print(f"    {pt}")

# === 8. Full Grid Generation Pipeline ===
def build_grid_from_keypoints(image, keypoints, row_eps=5, col_eps=3):
    img_copy = image.copy()
    pts = get_keypoint_coords(keypoints)

    # Cluster and fit horizontal lines
    row_labels = cluster_rows(pts, eps=row_eps)
    row_lines = fit_horizontal_lines(pts, row_labels)

    # Cluster and fit vertical lines
    col_labels = cluster_columns(pts, eps=col_eps)
    col_lines = fit_vertical_lines(pts, col_labels)

    # Print all points for each column cluster
    # print_column_clusters(pts, col_labels)

    # Sort row and column lines by their intercepts
    row_lines = sorted(row_lines, key=lambda line: line[1])  # Sort by intercept
    col_lines = sorted(col_lines, key=lambda line: line[1])  # Sort by intercept

    # Draw horizontal lines (red)
    draw_lines(img_copy, row_lines, x_range=(0, image.shape[1]), color=(0, 0, 255))

    # Draw vertical lines (green)
    draw_vertical_lines(img_copy, col_lines, y_range=(0, image.shape[0]), color=(0, 255, 0))

    return img_copy, row_lines, col_lines

def find_additional_grid_points(intersections, row_count, col_count, grid_size):
    new_points = []

    # Hardcode 3 new rows of grid points for top and bottom
    for i in range(col_count):
        ref_point = intersections[i + col_count * 1]
        for j in range(3):
            row_coordinate = ref_point.y + grid_size * (j + 1)
            col_coordinate = ref_point.x
            new_point = Point(col_coordinate, row_coordinate, j + 2, i)
            new_points.append(new_point)

    # Hardcode 2 new rows of grid points in the middle
    for i in range(col_count):
        ref_point = intersections[i + col_count * 6]
        for j in range(2):
            row_coordinate = ref_point.y + grid_size * (j + 1)
            col_coordinate = ref_point.x 
            new_point = Point(col_coordinate, row_coordinate, j + 7, i)
            new_points.append(new_point)

    # Bottom 
    for i in range(col_count):
        ref_point = intersections[i + col_count * 11]
        for j in range(3):
            row_coordinate = ref_point.y + grid_size * (j + 1)
            col_coordinate = ref_point.x 
            new_point = Point(col_coordinate, row_coordinate, j + 12, i)
            new_points.append(new_point)

    # Increment row indices for old points
    for i in range(col_count*5):
        intersections[i + col_count * 2].row_idx += 3
        intersections[i + col_count * 7].row_idx += 5

    for i in range(col_count*2):
        intersections[i + col_count * 12].row_idx += 8

    return np.concatenate((intersections, np.array(new_points)))


def grid_generation(og_image):
    image = og_image.copy()
    # Get image dimension
    height, width = image.shape[:2]

    # Set up SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = (height/200) ** 2
    #params.maxArea = 100
    params.filterByColor = True
    params.blobColor = 0  # 0 for dark blobs, 255 for light blobs
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.filterByConvexity = False
    params.filterByInertia = False

    # Create detector
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(image)

    # Draw detected keypoints
    im_with_keypoints = cv2.drawKeypoints(
        image, keypoints, np.array([]), (0, 255, 0),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )


    # Build grid
    grid_img, rows, cols = build_grid_from_keypoints(im_with_keypoints, keypoints, row_eps=height/100, col_eps=width/200)

    # Find intersections
    intersections = find_intersections(rows, cols)

    row_count = len(rows)
    col_count = len(cols)

    # Calculate grid size using by taking the average of the x differences between the first row of intersections
    grid_size = get_grid_size(row_count, col_count, intersections)
    print(f"Grid size (average x difference): {grid_size}")

    # Sort intersections by row_idx and col_idx
    intersections = sorted(intersections, key=lambda pt: (pt.row_idx, pt.col_idx))

    # Assume horizontal image orientation, insert more rows between the top 2 rails and the rest
    # Apply the same logic to create more points for the bottom as well
    intersections = find_additional_grid_points(intersections, row_count, col_count, grid_size)

    intersections = sorted(intersections, key=lambda pt: (pt.row_idx, pt.col_idx))

    for i in range(col_count*3):
        pt = intersections[i]
        (x, y) = pt.get_coordinates()
        # print(f"PT Coordinates {x}, {y} on row {pt.row_idx}, col {pt.col_idx}")


    # Draw intersection points
    for pt in intersections:
        (x, y) = pt.get_coordinates()
        cv2.circle(grid_img, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)

    # Make image smaller for display
    scale = 0.3
    new_size = (int(grid_img.shape[1] * scale), int(grid_img.shape[0] * scale))
    grid_img = cv2.resize(grid_img, new_size, interpolation=cv2.INTER_AREA)

    return grid_img, intersections, grid_size, row_count, col_count

# Check for if a grid point is covered by a Bit Mask
def get_masked_grid_points(grid_points, masks):
    masked_points = []
    for indx, item_mask in enumerate(masks):
        for gp in grid_points:
            (x, y) = gp.get_coordinates()
            if item_mask[int(round(y)), int(round(x))] > 0:
                masked_points.append(gp)

    return masked_points



num_classes = 3  # plugged, not_plugged, pass_over
input_size = 224

# Use the same transform as in training
data_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Run inference on each point in intersections and store results as GridPoint objects
def classify_snapshot(model, img):
    input_tensor = data_transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        _, pred = torch.max(output, 1)
    return pred.item(), probs.squeeze().cpu().numpy()


def classify_grid_points(colour_image, intersections, grid_size, height, width):

    # Load model from file

    # Define the model architecture (must match training)
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model.load_state_dict(torch.load("plug_classifier_resnet18.pth", map_location="cpu"))
    model.eval()

    # Run inference on each point in interesections and store results as GridPoint objects
    GridPoints = []

    half_grid_size = int(grid_size / 2) + 1  # Adjust grid size for cropping

    for pt in intersections:
        (x, y) = pt.get_coordinates()
        x = int(round(x))
        y = int(round(y))
        
        # Make sure the snapshot cannot go out of bound
        # Skip if the snapshot would go out of bounds
        if (y-half_grid_size < 0 or y+half_grid_size >= height or
            x-half_grid_size < 0 or x+half_grid_size >= width):
            continue

        snapshot = colour_image[y-half_grid_size:y+half_grid_size, x-half_grid_size:x+half_grid_size]
        # Convert NumPy array (OpenCV BGR) to PIL Image (RGB)
        snapshot_rgb = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(snapshot_rgb)
        pred, probs = classify_snapshot(model, pil_img)
        grid_point = GridPoint(pt, pred, probs)
        GridPoints.append(grid_point)

    return GridPoints


def visualise_classified_grid_points(image, GridPoints, scale=0.3):
    img_copy = image.copy()

    # Visualise results on the image, high light all plugged points
    for gp in GridPoints:
        (x, y) = gp.get_coordinates()
        if gp.type == 0:  # not_plugged
            cv2.circle(img_copy, (int(round(x)), int(round(y))), 10, (0, 0, 255), -1)
        elif gp.type == 1:  # pass_over
            cv2.circle(img_copy, (int(round(x)), int(round(y))), 10, (0, 255, 0), -1)
        else:  # plugged
            cv2.circle(img_copy, (int(round(x)), int(round(y))), 10, (255, 0, 255), -1)

    # Make image smaller for display
    new_size = (int(img_copy.shape[1] * scale), int(img_copy.shape[0] * scale))
    display_img = cv2.resize(img_copy, new_size,
                            interpolation=cv2.INTER_AREA)
    
    return display_img

