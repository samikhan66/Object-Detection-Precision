"""Unnormalizes the boundary boxes x1, y1, x2, y2 """

def unnormalize(box, h, w):
	assert len(box) == 4
	box = [
    	int(round(box[0]*h)),
    	int(round(box[1]*w)),
    	int(round(box[2]*h)),
    	int(round(box[3]*w))
    ]
	return box

def get_boundaries(List, height, width, boxes):
	return [unnormalize(boxes[x], height, width) for x in List]