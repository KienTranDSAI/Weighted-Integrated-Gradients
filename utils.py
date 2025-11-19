import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shap
import json 

URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
FNAME = shap.datasets.cache(URL)
with open(FNAME) as f:
    CLASS_NAMES = json.load(f)
NUM_IMAGES = 41
THRESHOLD = 0.25
NSAMPLE = 6
RSEED = 8


def convert_to_2channel(img):
   
    result = 0.2989 * img[0,:, :] + 0.5870 * img[1,:, :] + 0.1140 * img[2,:, :]
    return result

def normalize(image, mean = None, std = None):
    if mean == None:
       mean = [0.485, 0.456, 0.406]
    if std == None:
       std = [0.229, 0.224, 0.225]
    if image.max() > 1:
        image = image.astype(np.float64)
        image /= 255
    image = (image - mean) / std
    # in addition, roll the axis so that they suit pytorch
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()

def getTrueId(img, model, device = 'cpu'):
    model.to(device)
    img = (torch.from_numpy(img).permute(0,3,1,2)).to(device)
    y_pred = model(img)
    class_ = torch.argmax(y_pred, dim = 1)
    ans = torch.nn.functional.softmax(y_pred,1)
    return class_.item()

def get_sample_data(image_ind, images_path, masks_path, transform):
    image_path = images_path[image_ind]
    mask_path = masks_path[image_ind]
    image_raw_name = image_path.split("/")[-1].split(".")[0]

    # Load the image
    image = get_sample_image(image_path, transform)
    mask = get_sample_mask(mask_path, transform)

    return image_raw_name, image, mask



def get_sample_image(image_path, transform):
    image = Image.open(image_path)
    transformed_image = transform(image)
    return transformed_image
def get_sample_mask(mask_path, transform):
    mask = Image.open(mask_path)
    transformed_mask = transform(mask)
    return transformed_mask


def get_neutral_background(image):
    height, width = image.shape[:2]
    corner_size = int(0.1 * height)  # This will be 22 pixels for your 224x224 image
    top_left = image[:corner_size, :corner_size, :]
    top_right = image[:corner_size, -corner_size:, :]
    bottom_left = image[-corner_size:, :corner_size, :]
    bottom_right = image[-corner_size:, -corner_size:, :]
    average_top_left = np.mean(top_left, axis=(0, 1))
    average_top_right = np.mean(top_right, axis=(0, 1))
    average_bottom_left = np.mean(bottom_left, axis=(0, 1))
    average_bottom_right = np.mean(bottom_right, axis=(0, 1))
    average_all_corners = np.mean([average_top_left, average_top_right, average_bottom_left, average_bottom_right], axis=0)
    average_all_corners_broadcasted = average_all_corners[np.newaxis, np.newaxis, :]
    return average_all_corners_broadcasted  

def get_sorted_indices(val):
    flattened_array = val.flatten()
    sorted_indices = np.argsort(flattened_array)
    return sorted_indices

def get_top_k(val, k, sorted_indices = None):
    """
    Find top k biggest values
    """
    if sorted_indices is None:
      sorted_indices = get_sorted_indices(val)
    top_k_indices = sorted_indices[-k:]
    return top_k_indices

def create_mask_from_indices(val, k, sorted_indices = None):
    """
    Create a mask from the indices of the top k biggest values
    """
    top_k_indices = get_top_k(val, k, sorted_indices)
    mask = np.zeros_like(val, dtype=int)
    top_k_positions = np.unravel_index(top_k_indices, val.shape)
    mask[top_k_positions] = 1

    return mask

def get_raw_important_point(important_val, percentile):
    val = important_val
    threshold = np.percentile(important_val, percentile)
    indexes  = np.where(important_val > threshold)
    second_dim = indexes[0]
    third_dim = indexes[1]
    datapoint = [[second_dim[i],third_dim[i]] for i in range(len(second_dim))]
    datapoint = np.array(datapoint)
    return datapoint

def create_shap_image(shap_value, standard_threshold):
    important_point = get_raw_important_point(shap_value,standard_threshold)
    shap_image = np.zeros(shap_value.shape)
    for point in important_point:
        shap_image[point[0], point[1]] = 1
    return shap_image
 
def plot_saliency_with_topk(ax, saliency, original_image, title, k=500, cmap='hot', img_alpha=0.3, sali_alpha=0.6):

    """
    Creates a plot of saliency map showing only top-k values with the original image overlapped.
    
    Args:
        saliency: Array of saliency/importance values to visualize.
        original_image: The original image to overlay.
        title: Title for the saliency plot.
        k: Number of top values to display.
        cmap: Colormap to use for saliency visualization.
        alpha: Alpha value for the original image overlay (lower = more transparent).
    """
    # Create masked version showing only top-k values
    saliency = saliency.copy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    topk_saliency = np.zeros_like(saliency)
    
    # Get sorted indices
    flat_saliency = saliency.flatten()
    sorted_indices = np.argsort(flat_saliency)[::-1]  # Sort in descending order
    
    # Get top-k indices
    top_k_indices = sorted_indices[:k]
    
    # Create masked version with only top-k values
    flattened_topk = np.zeros_like(saliency.flatten())
    flattened_topk[top_k_indices] = flat_saliency[top_k_indices]
    topk_saliency = flattened_topk.reshape(saliency.shape)
    
    # Create figure with single plot
    # fig, ax = plt.subplots(figsize=(4, 4))
    # fig.patch.set_facecolor('black')
    
    # Set background color
    ax.set_facecolor('white')
    
    # First show the original image with low alpha
    ax.imshow(original_image, cmap='hot', alpha=img_alpha)
    
    # Then overlay the saliency map
    im = ax.imshow(topk_saliency, cmap=cmap, alpha=sali_alpha)
    
    # Set title and remove ticks
    ax.set_title(title, color='white', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add white border
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(2)

    plt.tight_layout()
    return ax

def get_score(model, device, image,img_indice = None):
	image = torch.from_numpy(image).permute(0,3,1,2)
	model.to(device)
	image = image.to(device)
	scores = model(image)
	if img_indice is None:
		img_indice = torch.argmax(scores, dim=1).tolist()
	softmax_scores = torch.nn.functional.softmax(scores, dim=1)
	return softmax_scores[0, img_indice]

def exact_find_d_alpha(model, device, to_explain,val,trueImageInd = None, target_ratio=0.5, neutral_val=0, epsilon=0.005, max_iter=100):
	low, high = 0, val.shape[0]*val.shape[1]
	sorted_indices = get_sorted_indices(val)
	full_score = get_score(model, device, to_explain, trueImageInd)
	# print(f"Full score: {full_score}")
	target_score = full_score * target_ratio
	iter_count = 0
	while high - low > 0 and iter_count < max_iter:
		iter_count += 1
		mid = int((low + high) / 2 )
		mask = create_mask_from_indices(val, mid, sorted_indices)
		mask = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
		background_mask = 1 - mask
		partial_image = to_explain[0]*background_mask + mask*neutral_val
		partial_image = np.expand_dims(partial_image, axis=0).astype(np.float32)
		score = get_score(model, device, partial_image, trueImageInd)
		if abs(score - target_score) < epsilon:
			break
		elif score > target_score:
			low = mid
		else:
			high = mid
	return mid, score 


###Deletion

def get_partial_score_batch(images, model, device, img_indices=None, class_names=CLASS_NAMES):
    # batch_images = torch.cat([torch.from_numpy(np.transpose(image, (2, 0, 1))).unsqueeze(0) for image in images], axis = 0)
    batch_images = torch.from_numpy(np.array(images)).permute(0,3,1,2)
    model.to(device)
    batch_images = batch_images.to(device)
    # print(batch_images.device)
    scores = model(batch_images)

    if img_indices is None:
        img_indices = torch.argmax(scores, dim=1).tolist()
    elif isinstance(img_indices, int):
        img_indices = [img_indices] * len(scores)

    softmax_scores = torch.nn.functional.softmax(scores, dim=1)
    softmax_scores_list = softmax_scores[torch.arange(len(img_indices)), img_indices].tolist()
    # softmax_scores_list = [softmax_scores[i, idx].item() for i, idx in enumerate(img_indices)]
    # img_names_list = [class_names[str(idx)][1] for idx in img_indices]

    # return img_names_list, softmax_scores_list
    return softmax_scores_list
def deletion_score_batch(to_explain, trueImageInd, val, model, device, percentile_list, neutral_val = 0, image_show = False):


  batch_partial_images = []
  for percentile in percentile_list:
    threshold = np.percentile(val, percentile)
    mask = val >= threshold
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
    background_mask = 1 - mask
    partial_image = to_explain[0]*background_mask + mask*neutral_val
    partial_image = partial_image.astype(np.float32)

    batch_partial_images.append(partial_image)
    if image_show:
      plt.imshow(partial_image)
      plt.show()
  return get_partial_score_batch(batch_partial_images, model, device, trueImageInd)
  # return batch_partial_images
def get_threshold_batch(val, percentile_area, x_values = None, step_size = 100, neutral_val = 0):

  if x_values == None:
    step_size_float = 100.0 / step_size
    x_values = np.arange(0, 100, step_size_float)
  else:
    x_values = np.array(x_values)
  percentile_list = 100 - x_values

  deletion_score_list  = deletion_score_batch(to_explain, trueImageInd, val, model, device, percentile_list, neutral_val)
  y_values = np.array(deletion_score_list)
  area_under_curve = np.trapz(y_values, x_values)
  print("Area under the curve:", area_under_curve)
  diff_x_values = np.diff(x_values)
  cumulative_area = np.cumsum(diff_x_values  * (y_values[:-1] + y_values[1:]) / 2)

  # Find the x-value where cumulative area is 50% of the total area
  target_area = percentile_area * area_under_curve
  print("Target area:", target_area)
  x_results = x_values[np.where(cumulative_area >= target_area)[0][0] + 1]
  print("x-result:", x_results)
  return x_results

def get_weight_batch(total_val, x_threshold, neutral_val):
  num_baseline = total_val.shape[0]
  weight_list = []
  for i in range(num_baseline):
    val = np.sum(total_val[i], axis = 0)
    weight_list.append(deletion_score_batch(to_explain, trueImageInd, val, model, device,[100 - x_threshold],neutral_val)[0])
  return weight_list
def get_point2remove(raw_shap_value, x_values = None, step_size = 100, ratio_score = 0.5, neutral_val = 0):
  if x_values == None:
    step_size_float = 50 / step_size
    x_values = np.arange(0, 30, step_size_float)
  else:
    x_values = np.array(x_values)
  percentile_list = 100 - x_values
  # print(percentile_list)
  deletion_score_list  = deletion_score_batch(to_explain, trueImageInd, raw_shap_value, model, device, percentile_list, neutral_val,False)
  # print(deletion_score_list)
  full_score = deletion_score_batch(to_explain, trueImageInd, raw_shap_value, model, device, [100], neutral_val)[0]
  for i in range(len(deletion_score_list)):
    if deletion_score_list[i]<=ratio_score*full_score:

      return x_values[i]
  return x_values[-1]

def get_auc_deletion(to_explain, trueImageInd, val, model, device, x_values = None, neutral_value = 0, image_show = False):
  if x_values is None:
    x_values = np.arange(0,100)
  raw_deletion_score_list = []
  # for i in range(0,100):
  # raw_deletion_score_list.append(deletion_score_batch(to_explain, trueImageInd, raw_shap_value, [100-i],average_all_corners_broadcasted)[0])
  raw_deletion_score_list = deletion_score_batch(to_explain, trueImageInd, val, model, device, x_values, neutral_value,image_show)
  y_values = np.array(raw_deletion_score_list)
  area_under_curve = np.trapz(y_values, x_values)
  return area_under_curve



import numpy as np

def find_d_alpha(to_explain, val, model, device, trueImageInd = None, target_ratio=0.5, neutral_val=0, epsilon=0.01, max_iter=100):
    """
    Binary search to find the proportion of pixels to remove so that the model score approaches target_score.

    Args:
        val (np.array): Input importance scores (e.g., IG attributions), shape (3,224,224)
        model (function): A function that takes an input image and returns a score
        target_score (float): The desired model score after removal
        neutral_val (int or float): The value to replace removed pixels
        epsilon (float): Convergence threshold
        max_iter (int): Maximum iterations for binary search

    Returns:
        float: The optimal removal percentage
    """
    low, high = 0, 100  # Search range in percentage
    iter_count = 0
    full_score = get_score(model, device, to_explain, trueImageInd)
    print(f"Full score: {full_score}")
    target_score = full_score * target_ratio
    score = full_score
    score_high = full_score
    score_low = 0
    while high - low > 0 and iter_count < max_iter:
        iter_count += 1
        mid = (low + high) / 2  # Current percentage to remove

        val_copy = val.copy()
        threshold = np.percentile(val_copy, mid)
        mask = val >= threshold
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
        background_mask = 1 - mask
        partial_image = to_explain[0]*background_mask + mask*neutral_val
        partial_image = np.expand_dims(partial_image, axis=0).astype(np.float32)

        print("SP: ", sum(partial_image!=0))
        score = get_score(partial_image, trueImageInd)
        # plt.figure(figsize=(5, 5))
        # plt.imshow(partial_image[0])  # Convert (3,224,224) to (224,224,3) for display
        # plt.title(f"Iteration {iter_count} - Removed: {mid:.2f}% - Score: {score:.4f}")
        # plt.axis("off")
        # plt.show()
        print(f"Score: {score}")
        if  abs(score - target_score) < epsilon:
          break
        elif score > target_score:
            score_high = score
            print(f"High score: {score_high}, low_score: {score_low}")
            print(f"High: {high},Mid: {mid}, Low: {low}")

            high = mid  # Reduce removal percentage

        else:
            score_low = score
            print(f"High score: {score_high}, low_score: {score_low}")
            print(f"High: {high},Mid: {mid}, Low: {low}")
            low = mid   # Increase removal percentage


    return mid, score
# def exact_find_d_alpha(to_explain,val,trueImageInd = None, target_ratio=0.5, neutral_val=0, epsilon=0.005, max_iter=100):
#   low, high = 0, val.shape[0]*val.shape[1]
#   sorted_indices = get_sorted_indices(val)
#   full_score = get_score(to_explain, trueImageInd)
#   # print(f"Full score: {full_score}")
#   target_score = full_score * target_ratio
#   iter_count = 0
#   while high - low > 0 and iter_count < max_iter:
#     iter_count += 1
#     mid = int((low + high) / 2 )
#     mask = create_mask_from_indices(val, mid, sorted_indices)
#     mask = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
#     background_mask = 1 - mask
#     partial_image = to_explain[0]*background_mask + mask*neutral_val
#     partial_image = np.expand_dims(partial_image, axis=0).astype(np.float32)
#     score = get_score(partial_image, trueImageInd)

#     # plt.figure(figsize=(5, 5))
#     # plt.imshow(partial_image[0])  # Convert (3,224,224) to (224,224,3) for display
#     # plt.title(f"Iteration {iter_count} - Removed: {mid:.2f}% - Score: {score:.4f}")
#     # plt.axis("off")
#     # plt.show()

#     # print(f"Score: {score}, high: {high}, low: {low}, mid: {mid}")
#     if  abs(score - target_score) < epsilon:
#       break
#     elif score > target_score:
#       low = mid  # Reduce removal percentage
#     else:
#       high = mid   # Increase removal percentage
#   return mid,score

import copy
import numpy as np
import matplotlib.pyplot as plt

def filter_weights(weights, threshold):
  num_remove = 0
  weights_array = np.array(weights)
  mean_weight = np.mean(weights_array)
  imax = np.argmax(weights_array)
  for i in range(len(weights_array)):
    if weights[imax] < mean_weight * threshold:
      weights_array = np.zeros_like(weights_array)
      weights_array[imax] = weights[imax]
      num_remove = len(weights_array) - 1
    elif weights_array[i] < threshold * mean_weight:
      weights_array[i] = 0
      num_remove += 1

  return np.array(weights_array / np.sum(weights_array)), num_remove