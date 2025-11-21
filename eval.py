import os

from torchvision import models
from torchvision import transforms

from explainer import IGExplainer
from utils import *
import random
import torch
import numpy as np
rseed = 8
torch.manual_seed(rseed)
np.random.seed(rseed)
torch.cuda.manual_seed(rseed)
torch.cuda.manual_seed_all(rseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_ig_explanation(image_ind, img_paths, mask_paths, transform, model, device, local_smoothing = 0):
	shap_value_list = []
	grad_list = []
	local_smoothing = 0
	img_path, mask_path = img_paths[image_ind], mask_paths[image_ind]
	transformed_image, transformed_mask = get_sample_image(img_path, transform), get_sample_mask(mask_path, transform)
	to_explain = np.array(transformed_image.permute(1,2,0).unsqueeze(0))
	white_baselie = np.ones(to_explain.shape)
	black_baseline = np.zeros(to_explain.shape)
	# black_baseline1 = np.zeros(to_explain.shape)
	median_baseline = np.ones(to_explain.shape)*0.5
	raw_image_baseline = np.array(get_sample_data(0, img_paths, mask_paths, transform)[1].permute(1,2,0).unsqueeze(0))
	random_baseline = np.random.rand(*to_explain.shape)
	random_baseline1 = np.random.rand(*to_explain.shape)
	baseline = np.concatenate([black_baseline, raw_image_baseline, random_baseline,random_baseline1, white_baselie,median_baseline], axis = 0)
	# baseline = np.concatenate([itself_baseline, white_baselie, black_baseline, random_baseline], axis = 0)
	# baseline = np.concatenate([white_baselie, black_baseline, random_baseline,black_baseline1], axis = 0)
	trueImageInd = getTrueId(to_explain, model, device)
	average_all_corners_broadcasted = get_neutral_background(to_explain[0])
	normalized_baseline = normalize(baseline).to(device)
	explainer = IGExplainer(model.to(device), normalized_baseline, local_smoothing = local_smoothing)
	
	#Get gradient from shapleyvalue
	shap_values, indexes, baseline_samples, individual_grads = explainer.shap_values(
		normalize(to_explain).to(device), ranked_outputs=1, nsamples = nsamples, rseed = rseed)
	shap_values = [np.swapaxes(s, 0, -1) for s in shap_values]
	raw_shap_value = np.sum(shap_values[0], axis = (0,-1)) #Raw shap values

	return raw_shap_value, individual_grads



def eval(img_paths, mask_paths, transform, model, device, local_smoothing = 0):
	weighted_baseline_value_list = []
	shap_deletion_score = []
	weighted_baseline_deletion_score = []
	filtered_weighted_baseline_deletion_score = []
	list_num_remove = []

	weights_array = []
	filtered_weights_array = []

	rseed = 8
	TOTAL_IMAGES = len(img_paths)
	random.seed(rseed)
	image_indices = list(range(TOTAL_IMAGES))
	random.shuffle(image_indices)

	for ind in range(NUM_IMAGES):
		image_ind = image_indices[ind]
		print(f"Image path: {img_paths[image_ind]}")
		image_raw_name, transformed_image, transformed_mask = get_sample_data(image_ind, img_paths, mask_paths, transform)
		to_explain = np.array(transformed_image.
						permute(1,2,0).unsqueeze(0))
		baseline = np.zeros(to_explain.shape)
		trueImageInd = getTrueId(to_explain, model.to('cpu'))
		average_all_corners_broadcasted = get_neutral_background(to_explain[0])
		raw_shap_value, individual_grads = get_ig_explanation(image_ind, img_paths, mask_paths, transform, model, device)
		weight_list = []
		score_list = []
		for ind in range(len(individual_grads)):
			individual_grad = individual_grads[ind]
			# individual_grad = np.abs(individual_grads[ind])
			individual_val = np.sum(individual_grad, axis = 0)
			num_of_deleted_point, score = exact_find_d_alpha( model, device, to_explain, individual_val, trueImageInd = trueImageInd, target_ratio=0.5, neutral_val=average_all_corners_broadcasted, epsilon=0.005, max_iter=100)
			# print(f"num_of_deleted_point: {num_of_deleted_point}, score: {score}")
			weight_list.append(num_of_deleted_point)
			score_list.append(score)
			# print("---------------------------")
		weight_list = np.array(weight_list)
		# print(weight_list)
		# weight_list = (1/(1 - weight_list/50176))/(1/(1 - weight_list/50176)).sum()
		weight_list = (50176/weight_list)/(50176/weight_list).sum()

		print(f"Image id: {image_ind}")
		print(f"weight_list: {weight_list}")
		weights_array.append(weight_list)

		filtered_weight_list, num_remove = filter_weights(weight_list, THRESHOLD)
		list_num_remove.append(num_remove)

		print(f"filtered_weight_list: {filtered_weight_list}")
		filtered_weights_array.append(filtered_weight_list)
		weight_list = weight_list.reshape(-1,1,1,1)
		filtered_weight_list = filtered_weight_list.reshape(-1,1,1,1)

		weighted_basedline_shap_val = np.sum(weight_list * individual_grads, axis = (0,1))
		filtered_weighted_basedline_shap_val = np.sum(filtered_weight_list * individual_grads, axis = (0,1))
		weighted_baseline_value_list.append(weighted_basedline_shap_val)

		shap_area_under_curve = get_auc_deletion(to_explain, trueImageInd, raw_shap_value, model, device, x_values = None, neutral_value = average_all_corners_broadcasted)
		weighted_baseline_area_under_curve = get_auc_deletion(to_explain, trueImageInd, weighted_basedline_shap_val, model, device, x_values = None, neutral_value = average_all_corners_broadcasted)
		filtered_weighted_baseline_area_under_curve = get_auc_deletion(to_explain, trueImageInd, filtered_weighted_basedline_shap_val, model, device, x_values = None, neutral_value = average_all_corners_broadcasted)
		shap_deletion_score.append(shap_area_under_curve)
		weighted_baseline_deletion_score.append(weighted_baseline_area_under_curve)
		filtered_weighted_baseline_deletion_score.append(filtered_weighted_baseline_area_under_curve)

	return shap_deletion_score, weighted_baseline_deletion_score, filtered_weighted_baseline_deletion_score

if __name__ == "__main__":
	#img_paths = [os.path.join("./data/Image", i) for i in  os.listdir("./data/Image")]
	#mask_paths = [os.path.join("./data/Image", i) for i in os.listdir("./data/Mask")]
	dataset_dir = "/kaggle/input/old-data/Adaptive"
	images_raw_names = [i for i in os.listdir(dataset_dir + "/Image")]
	img_paths = [dataset_dir + f"/Image/{name}" for name in images_raw_names]
	mask_paths = [dataset_dir + f"/Mask/{name}" for name in images_raw_names]
	# ind = 5
	# img_path, mask_path = img_paths[ind], mask_paths[ind]
	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor()
	])
	model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# val, grads = get_ig_explanation(img_path, mask_path, transform, model, device)
	# print(grad.shape)
	shap_deletion_score, weighted_baseline_deletion_score, filtered_weighted_baseline_deletion_score = eval(img_paths, mask_paths, transform, model, device, local_smoothing = 0)
