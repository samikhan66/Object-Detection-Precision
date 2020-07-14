""" PRECISION SCRIPT USED TO MEASURE PRECISION OF COMMON OBJECT DETECTION ITEMS 
Written by Sami Khan
February 2020
ALL RIGHTS RESERVED
 """


import os
import cv2
import xml.etree.ElementTree as et
import test_ml
import statistics
import time
import numpy as np
from collections import namedtuple, defaultdict
import shutil

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def get_attribute_list(attribute_object):
	"""Finds attribute coordinates from the dictionary for every item
	Args: Takes in the dictionary that contains attributes along with their boundary boxes
	Returns: A list containing the xmin, xmax, ymin, ymax of that annotated object
	"""
	return [attribute_object.find('bndbox/ymin').text, 
	attribute_object.find('bndbox/xmin').text,
	attribute_object.find('bndbox/ymax').text,
    attribute_object.find('bndbox/xmax').text]


def parse_xml(files, ATTRIBUTES, SINGLE_ATTRIBUTES_LIST):
	"""Finds the boundary boxes of different annotations
	Args: XML file that needs to be parsed
	Returns: A dictionary containing attributes along with their boundary boxes
	"""
	true_dic = {}
	try:
		tree = et.parse(files)
		root = tree.getroot()
		try:
			for attribute in ATTRIBUTES:
				if attribute in SINGLE_ATTRIBUTES_LIST:
					for attribute_object in root.findall(f"./object[name='{attribute}']"):
						if not attribute in true_dic:
							true_dic[attribute] = [get_attribute_list(attribute_object)]
						else:
							true_dic[attribute].append(get_attribute_list(attribute_object))
				else:
					for attribute_object in root.findall(f"./object[name='{attribute}']"):
						if not attribute in true_dic:
							true_dic[attribute] = [get_attribute_list(attribute_object)]
						else:
							true_dic[attribute].append(get_attribute_list(attribute_object))
		except Exception as e:
			print("1")
			print("1",e)
	except Exception as e:
		print(e) 
	return true_dic                        

def get_overlap(true_box, pred_box):
	"""Finds the overlap of the predicted box and the annotated box and the maximum area using that overlap
	Args: True box(annotated box) and predicted box(from the inference graph)
	Returns: Overlap and maximum area containing the boxes
	"""
	xT1, yT1, xT2, yT2 = true_box
	xP1, yP1, xP2, yP2 = pred_box
	max_area = (max((int(xP2) - int(xT1))*(int(yT2)-int(yT1)),
	 (int(xP2) - int(xP1))*(int(yP2)-int(yP1))))
	area_true = Rectangle(int(xT1), int(yT1), int(xT2), int(yT2))
	area_predicted = Rectangle(int(xP1), int(yP1), int(xP2), int(yP2))
	overlap = area(area_true, area_predicted)	
	return overlap, max_area

def calculate_single_precision(Rectangle, true_dic, predicted_dic, attribute):
	"""Finds the precision using the simple formula (precision = ((max_area - (max_area-overlap)) / max_area) * 100)
	Goes through different conditions to determine the precision accordingly 
	Args: Rectangle(tuple), dictionaries containing actual and predicted items, and the attribute that needs to be parsed
	Returns: Preicison
	"""
	precision_list = []
	if attribute in true_dic and attribute in predicted_dic:
		for i, coords in enumerate(true_dic[attribute]):
			overlap, max_area = (get_overlap(true_dic[attribute][i],
								 predicted_dic[attribute][0]))
			if overlap:
				precision = round(((max_area - (max_area-overlap)) / max_area) * 100, 1)
				precision_list.append(precision)
			else:
				precision_list.append(0) 
		return max(precision_list)	   
	elif ((attribute in true_dic and attribute not in predicted_dic) or
	(attribute in predicted_dic and attribute not in true_dic)):
		precision = 0
	else:
		precision = 100
	
	return precision

def one_true_one_pred_mult_item(true_dic, predicted_dic, attribute):
	"""Finds precision if only one bottle/wine glass is present """
	overlap, max_area = (get_overlap(true_dic[attribute][0],
	 predicted_dic[attribute][0]))
	if overlap:
		precision = round(((max_area - (max_area-overlap)) / max_area) * 100, 1)
	else:
		precision = 0 
	return precision			
	
def two_true_two_pred_mult_item(true_dic, predicted_dic, true_values_dic, 
	position_taken, item_0_list, item_1_list, true_0_dic, true_1_dic, attribute):
	"""Finds precision if two bottle/wine glass are present by 
	   pairing overlaps to the attribute, pairing it to which it has a maximum
	   overlap with.
	   If the overlap's position is taken, the next most maximum overlap is assigned to the other part
	   of the attribute """
	for key, value in true_0_dic.items():
		item_0_list.append(0 if value is None else value)
	for key, value in true_1_dic.items():
		item_1_list.append(0 if value is None else value)
	max_first = max(item_0_list)
	index_first = np.squeeze([i for i, j in enumerate(item_0_list)
								 if j == max_first])
	position_taken[str(index_first)] = max_first
	max_second = max(item_1_list)
	index_second = np.squeeze([i for i, j in enumerate(item_0_list)
								 if j == max_first])
	if str(index_second) not in position_taken:
		position_taken[str(index_second)] = max_second
	elif str(index_first) == '0':
		position_taken['1'] = max_second	
	else:
		position_taken['0'] = max_second		
	return list(position_taken.values())

def one_true_two_pred_mult_item(true_dic, predicted_dic, true_values_dic, 
	position_taken, item_0_list, item_1_list, true_0_dic, attribute):
	"""Finds precision if two bottle/wine glass are present by 
		   pairing overlaps to the attribute, pairing it to which it has a maximum
		   overlap with.
		   If the overlap's position is taken, the next most maximum overlap is assigned to the other part
		   of the attribute """
	for key, value in true_0_dic.items():
		item_0_list.append(0 if value is None else value)
	max_first = max(item_0_list)
	index_first = np.squeeze([i for i, j in enumerate(item_0_list)
								 if j == max_first])
	position_taken[str(index_first)] = max_first
	if '1' not in position_taken:
		position_taken['1'] = 0
	else:
		position_taken['0'] = 0	

	return list(position_taken.values())

def two_true_one_pred_mult_item(true_dic, predicted_dic, true_values_dic, 
	position_taken, item_0_list, item_1_list, true_0_dic, true_1_dic, attribute):
	"""Finds precision if bottle/wine glass are present by 
	   pairing overlaps to the attribute, pairing it to which it has a maximum
	   overlap with.
	   If the overlap's position is taken, the next most maximum overlap is assigned to the other part
	   of the attribute """
	for key, value in true_0_dic.items():
		item_0_list.append(0 if value is None else value)
	for key, value in true_1_dic.items():
		item_1_list.append(0 if value is None else value)	
	max_first = max(item_0_list)
	index_first = np.squeeze([i for i, j in enumerate(item_0_list) 
								if j == max_first])
	position_taken[str(index_first)] = max_first
	max_second = max(item_1_list)
	index_second = np.squeeze([i for i, j in enumerate(item_0_list) 
								if j == max_first])
	if str(index_second) not in position_taken:
		position_taken[str(index_second)] = 0
	elif str(index_first) == '0':
		position_taken['1'] = 0	
	else:
		position_taken['0'] = 0	
	position_taken['1'] = 0	

	return list(position_taken.values())

def multiple_items(true_dic, predicted_dic, attribute):
	"""Finds the precision at each position of the overlap if multiple line items exist """
	true_values_dic = {}
	position_taken = {}
	item_0_list = []
	item_1_list = []
	for i, true_box in enumerate(true_dic[attribute]):
		for j, pred_box in enumerate(predicted_dic[attribute]):
			overlap, max_area  = get_overlap(true_box, pred_box)
			if overlap is not None:
				precision = round(((max_area - (max_area-overlap)) / max_area) * 100, 1)
				if not "true_" + str(i) in true_values_dic:
					true_values_dic["true_" + str(i)] = {str(j): precision}
				else:
					true_values_dic["true_"+str(i)][str(j)] = precision
	if 	predicted_dic[attribute] is not None:			
		if len(true_dic[attribute]) == 2 and len(predicted_dic[attribute]) == 2:
			true_0_dic = true_values_dic['true_0']
			true_1_dic = true_values_dic['true_1']
			position_taken = two_true_two_pred_mult_item(true_dic, predicted_dic,
				true_values_dic, position_taken, item_0_list, item_1_list, 
				true_0_dic, true_1_dic, attribute)		
		elif len(true_dic[attribute]) == 1 and len(predicted_dic[attribute]) == 2:
			true_0_dic = true_values_dic['true_0']
			position_taken = one_true_two_pred_mult_item(true_dic, predicted_dic,
			 true_values_dic, position_taken, item_0_list, item_1_list, 
			 true_0_dic, attribute)
		elif len(true_dic[attribute]) == 2 and len(predicted_dic[attribute]) == 1:
			if 'true_0'in true_values_dic and 'true_1' in true_values_dic:
				true_0_dic = true_values_dic['true_0']
				true_1_dic = true_values_dic['true_1']
			elif 'true_0' not in true_values_dic:
				true_1_dic = true_values_dic['true_1']
				if  '1' in true_1_dic:
					true_values_dic['true_0'] = {'0': 0}	
					true_0_dic = true_values_dic['true_0']

				else:
					true_values_dic['true_0'] = {'1': 0}	
					true_0_dic = true_values_dic['true_0']		
			else:
				if '1' in true_0_dic:
					true_values_dic['true_1'] = {'0': 0}		
					true_1_dic = true_values_dic['true_0']
				else:
					true_values_dic['true_1'] = {'1': 0}	
					true_1_dic = true_values_dic['true_1']				
			position_taken = two_true_one_pred_mult_item(true_dic, predicted_dic,
			 true_values_dic, position_taken, item_0_list, item_1_list,
			  true_0_dic, true_1_dic, attribute)
				
	return position_taken

def calculate_mult_precision(Rectangle, true_dic, predicted_dic, attribute):
	"""Finds the precision using the simple formula (precision = ((max_area - (max_area-overlap)) / max_area) * 100)
	Goes through different conditions to determine the precision accordingly 
	Args: Rectangle(tuple), dictionaries containing actual and predicted items, and the attribute that needs to be parsed
	Returns: Preciison
	"""
	if predicted_dic[attribute] is not None:
		if attribute in true_dic and attribute in predicted_dic:
			if len(true_dic[attribute]) == 1 and len(predicted_dic[attribute]) == 1:
				precision = one_true_one_pred_mult_item(true_dic, predicted_dic, attribute)
			elif ((len(true_dic[attribute]) and len(predicted_dic[attribute])) == 2 or
			(len(true_dic[attribute]) == 1 and len(predicted_dic[attribute])) == 2 or
			(len(true_dic[attribute]) == 2 and len(predicted_dic[attribute])) == 1):
				precision = multiple_items(true_dic, predicted_dic, attribute)
				return np.squeeze(precision)

		elif ((attribute in true_dic and attribute not in predicted_dic) or
		(attribute in predicted_dic and attribute not in true_dic)):
			precision = 0
		else:
			precision = 100
		return [precision]

def compare_coordinates(SINGLE_ATTRIBUTES_LIST, true_dic, predicted_dic):
	"""Calculates precision for every attribute"""	
	precision_dic = {}
	for attribute in predicted_dic.keys():
		if attribute in SINGLE_ATTRIBUTES_LIST:
			precision = calculate_single_precision(Rectangle,
						true_dic, predicted_dic, attribute)
			precision_dic[attribute] = precision
		else:
			precision = calculate_mult_precision(Rectangle,
						true_dic, predicted_dic, attribute)
			precision_dic[attribute] = precision	

	return precision_dic		
                   

def area(a, b):
	"""Finds the area overlap between the two rectangles
	returns None if rectangles don't intersect
	"""
	dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
	dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
	if (dx>=0) and (dy>=0):
		return dx*dy

def write_csv_one_bottle_one_vase(files, precision_dic):
	return {"filename": files,
		"cup_precision": precision_dic["cup"],
		"laptop_precision": precision_dic["laptop"],
		"wine_glass_1_precision": precision_dic["wine_glass"][0],
		"wine_glass_2_precision": np.nan,
		"handbag_1_precision": precision_dic["handbag"][0],
		"handbag_2_precision": np.nan,
		"dining_table_precision": precision_dic["dining_table"],
		"handbag_precision": precision_dic["handbag"],
		"mouse_precision": precision_dic["mouse"],
		"vase_precision": precision_dic["vase"]}

def write_csv_two_bottle_one_vase(files, precision_dic):
	return {"filename": files,
					"cup_precision": precision_dic["cup"],
					"laptop_precision": precision_dic["laptop"],
					"wine_glass_1_precision": precision_dic["wine_glass"][0],
					"wine_glass_2_precision": precision_dic["wine_glass"][1],
					"handbag_1_precision": precision_dic["handbag"][0],
					"handbag_2_precision": np.nan,
					"dining_table_precision": precision_dic["dining_table"],
					"handbag_precision": precision_dic["handbag"],
					"mouse_precision": precision_dic["mouse"],
					"vase_precision": precision_dic["vase"]}

def write_csv_two_bottle_two_vase(files, precision_dic):
	return {"filename": files,
					"cup_precision": precision_dic["cup"],
					"laptop_precision": precision_dic["laptop"],
					"wine_glass_1_precision": precision_dic["Line_item"][0],
					"wine_glass_2_precision": precision_dic["Line_item"][1],
					"handbag_1_precision": precision_dic["handbag"][0],
					"handbag_2_precision": precision_dic["handbag"][1],
					"dining_table_precision": precision_dic["dining_table"],
					"handbag_precision": precision_dic["handbag"],
					"mouse_precision": precision_dic["mouse"],
					"vase_precision": precision_dic["vase"]}	

											

