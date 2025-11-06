import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2.data import transforms as T
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances

from detectron2.utils.visualizer import ColorMode
from PIL import Image
import matplotlib.pyplot as plt
from Mask import extract_instance
from ResistorProcessing import ResistorProcessingResult, ResistorResult, image_horitzontal_alignment, remove_glare, extract_resistance_from_predictor_output
from Grid import GridPoint, grid_generation, get_masked_grid_points, classify_grid_points, visualise_classified_grid_points


from GridDijkstra import GridDijkstraSolver, GridDijkstraSolver

######################################################################
#
#
# SETUP DETECTRON2 MODELS AND DATASETS
#
#
######################################################################

register_coco_instances("my_dataset_train_seg", {}, "train_coco.json", "dataset/train")

# Now, get the metadata
resistor_meta = MetadataCatalog.get("my_dataset_train_seg")

register_coco_instances("my_dataset_train_detect", {}, "extracted_coco.json", "extracted_images")

# Now, get the metadata
colour_bands_meta = MetadataCatalog.get("my_dataset_train_detect")

# populate thing_classes from COCO json categories (fallback to manual list if absent)
json_path = os.path.join("extracted_coco.json")
with open(json_path, "r", encoding="utf-8") as f:
    coco = json.load(f)
categories = [c.get("name") for c in coco.get("categories", [])]

if categories:
    colour_bands_meta.thing_classes = categories
else:
    # fallback — must match the order used during training
    colour_bands_meta.thing_classes = [
        "black","brown","red","orange","yellow","green","blue","violet","grey","white","gold"
    ]

cfg_seg = get_cfg()
cfg_seg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg_seg.DATASETS.TRAIN = ("my_dataset_train_seg")
cfg_seg.DATASETS.TEST = ()
cfg_seg.DATALOADER.NUM_WORKERS = 2
cfg_seg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg_seg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg_seg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg_seg.SOLVER.MAX_ITER = 500    # 500 iterations
cfg_seg.SOLVER.STEPS = []        # do not decay learning rate
cfg_seg.MODEL.DEVICE = "cpu"   # <--- force CPU
cfg_seg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg_seg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

cfg_detect = get_cfg()
cfg_detect.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg_detect.DATASETS.TRAIN = ("my_dataset_train_detect")
cfg_detect.DATASETS.TEST = ()
cfg_detect.DATALOADER.NUM_WORKERS = 2
cfg_detect.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg_detect.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg_detect.SOLVER.BASE_LR = 0.00005  # pick a good LR
cfg_detect.SOLVER.MAX_ITER = 2000    # 2000 iterations
cfg_detect.SOLVER.STEPS = []        # do not decay learning rate
cfg_detect.MODEL.DEVICE = "cpu"   # <--- force CPU
cfg_detect.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg_detect.MODEL.ROI_HEADS.NUM_CLASSES = 11

seg_model_path = "E:/UNSW/Thesis/outPuts/seg/model_final3.pth"

cfg_seg.MODEL.WEIGHTS = seg_model_path  # path to the model we have uploaded/downloaded
cfg_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75   # set a custom testing threshold
predictor_seg = DefaultPredictor(cfg_seg)

detection_model_path = "E:/UNSW/Thesis/outPuts/bands/model_final2.pth"

cfg_detect.MODEL.WEIGHTS = detection_model_path  # path to the model we have uploaded/downloaded
cfg_detect.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75   # set a custom testing threshold
predictor_detect = DefaultPredictor(cfg_detect)


#######################################################################
# Global Variables
#######################################################################

detected_grid_points: list[GridPoint] = []  

# Hardcoded list of Voltage supply nodes
# Assume Positive supply on top rail
top_power_supply_node_ids = (999,1000)
bottom_power_supply_node_ids = (1001,1002)

#######################################################################
# MAIN LOGIC TO RUN SEGMENTATION AND DETECTION
#######################################################################

async def run_segmentation_on_image(im):
    """
    Take an input image in CV2 format, run the segmentation model to extract resistors,
    then run the detection model on each extracted resistor to get colour bands and calculate resistance.

    Args:
        im: Input image in CV2 format (BGR)
    Returns:
        Tuple:
            - seg_out_img: Image showing segmentation results,
            - class_grid_im: Image showing classified grid points,
            - res_results: List of ResistorProcessingResult objects for each detected resistor
    """
    
    # Run the segmentation model
    outputs = predictor_seg(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v_seg = Visualizer(im[:, :, ::-1],
                    metadata=resistor_meta,
                    scale=1,
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v_seg.draw_instance_predictions(outputs["instances"].to("cpu"))
    seg_out_img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)


    # Get the masks
    masks = np.asarray(outputs["instances"].pred_masks.to("cpu"))


    res_results = []

    # for each mask, extract the instance and process it, then run detection model on it
    for i, item_mask in enumerate(masks):

        extracted_image = extract_instance(item_mask, im)

        rotated = image_horitzontal_alignment(np.array(extracted_image))

        image_without_glare = remove_glare(rotated)

        outputs_detect = predictor_detect(image_without_glare)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

        resistance = 0
        # calculate resistance
        try:
            resistance = extract_resistance_from_predictor_output(outputs_detect)

        except Exception as e:
            print(f"Error extracting resistance: {e}")

        v_detect = Visualizer(image_without_glare[:, :, ::-1],
                        metadata=colour_bands_meta,
                        scale=2,
                        instance_mode=ColorMode.IMAGE
        )
        detection_out = v_detect.draw_instance_predictions(outputs_detect["instances"].to("cpu"))
        res_result = ResistorProcessingResult(
            resistance=resistance,
            image=cv2.cvtColor(detection_out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB),
            id=i
        )

        res_results.append(res_result)

    # Run Grid generation
    grid_img, intersections, grid_size, row_count, col_count = grid_generation(im)
    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)


    # dilate the masks to cover larger area
    size = int(grid_size/4) * 2 + 1
    print(size)
    kernel = np.ones((size,size),np.uint8)
    masks_dilated = []
    for mask in masks:
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        masks_dilated.append(dilated_mask.astype(bool))


    # get image dimensions
    height, width = im.shape[:2]

    global detected_grid_points
    detected_grid_points = classify_grid_points(im, intersections, grid_size, height, width)

    # Show all mask covered grid points
    _ = get_masked_grid_points(detected_grid_points, masks_dilated)

    class_grid_im = visualise_classified_grid_points(im, detected_grid_points)
    class_grid_im = cv2.cvtColor(class_grid_im, cv2.COLOR_BGR2RGB)
    
    # return row_count as well so callers (UI) can run dijkstra correctly
    return seg_out_img, class_grid_im, res_results, row_count


async def run_dijkstra_on_grid(resistors: list[ResistorProcessingResult], row_count: int) -> list[ResistorResult]:
    """
    Run Dijkstra on Grid
    """

    # Check detected_grid_points is not empty, raise Exception if it is
    global detected_grid_points
    if not detected_grid_points:
        raise Exception("No grid points detected. Please run segmentation first.")

    all_plugged_gps = [gp for gp in detected_grid_points if gp.is_plugged()]

    # Run Dijkstra Solver
    dijkstra_solver = GridDijkstraSolver(detected_grid_points)

    results = dijkstra_solver.run_dijkstra()

    # reset gp node_ids
    for gp in all_plugged_gps:
        gp.node_id = None

    # For each resistor (masked_gp), find two plugged_gp with the lowest cost

    # List of ResistorResult
    rest_results = []
    unique_plugged_gps_list = []

    for i, res_p_result in enumerate(resistors):
        corresponding_results = [res for res in results if res.masked_gp_id == i]
        # sort the results by cost
        corresponding_results.sort(key=lambda x: x.total_cost)

        # Select the top 2 results with the lowest cost
        top_results = corresponding_results[:2]

        if len(top_results) >= 2:
            rest_result = ResistorResult(resistance=res_p_result.resistance, id=i, plugged_gps=[top_results[0].plugged_gp, top_results[1].plugged_gp])
            rest_results.append(rest_result)
            print(rest_result)
        else:
            print(f"Not enough Dijkstra results for masked_gp Resistor {i}")

    # Collect all unique plugged grid points
    unique_plugged_gps = {}
    for res in rest_results:
        for gp in res.plugged_gps:
            unique_plugged_gps[gp.get_index()] = gp  # using a dictionary to avoid duplicates
    unique_plugged_gps_list = list(unique_plugged_gps.values())

    # Find all the plugged GPs that not belong to any resistor connections
    non_res_plugged_gps = [gp for gp in all_plugged_gps if gp not in unique_plugged_gps_list]

    # Find all wire connected plugged GP pairs
    wire_connected_gp_pairs = dijkstra_solver.find_wire_connected_gps(non_res_plugged_gps)
    print(wire_connected_gp_pairs)

    # If the GPs are in row 5-9 and in same col, they can be treated as the same point,
    # likewise if the GPs are in row 12-16 and in same col, they can be treated as the same point.
    # Assign node ids to unique plugged grid points
    node_id = 1
    for gp in unique_plugged_gps_list:
        if gp.node_id is not None:
            continue  # already assigned
        if gp.get_index()[0] in range(5, 10):
            gp.node_id = node_id
            # assign same node_id to all GPs in the same col and row 5-9
            for other_gp in all_plugged_gps:
                if other_gp.get_index()[1] == gp.get_index()[1] and other_gp.get_index()[0] in range(5, 10):
                    other_gp.node_id = node_id
            node_id += 1
        elif gp.get_index()[0] in range(12, 17):
            gp.node_id = node_id
            # assign same node_id to all GPs in the same col and row 12-16
            for other_gp in all_plugged_gps:
                if other_gp.get_index()[1] == gp.get_index()[1] and other_gp.get_index()[0] in range(12, 17):
                    other_gp.node_id = node_id
            node_id += 1
        else:
            gp.node_id = node_id
            node_id += 1


    # Hardcoded list of Voltage supply nodes
    # Assume Positive supply on top rail
    global top_power_supply_node_ids,bottom_power_supply_node_ids

    # Top and bottom power rails, find out if any plugged GPs are on the power rail, get their node ids, assign them the same node id
    top_rail_v_node = set()
    top_rail_gnd_node = set()
    bottom_rail_v_node = set()
    bottom_rail_gnd_node = set()
    for gp in unique_plugged_gps_list:
        if gp.get_index()[0] == 0:  # top power rail row index
            top_rail_v_node.add(gp)
        elif gp.get_index()[0] == 1:  # top gnd rail row index
            top_rail_gnd_node.add(gp)
        elif gp.get_index()[0] == row_count - 2:  # bottom gnd rail row index
            bottom_rail_gnd_node.add(gp)
        elif gp.get_index()[0] == row_count - 1:  # bottom power rail row index
            bottom_rail_v_node.add(gp)

    # Assign same node id to all top rail V nodes

    for gp in top_rail_v_node:
        gp.node_id = top_power_supply_node_ids[0]
    for gp in top_rail_gnd_node:
        gp.node_id = top_power_supply_node_ids[1]
    for gp in bottom_rail_gnd_node:
        gp.node_id = bottom_power_supply_node_ids[0]
    for gp in bottom_rail_v_node:
        gp.node_id = bottom_power_supply_node_ids[1]


    # For all pairs of GPs present in wire_connected_gp_pairs, assign them the same node id
    for gp1, gp2 in wire_connected_gp_pairs:
        # Check both have node ids
        if gp1.node_id is None or gp2.node_id is None:
            continue
        # assign the smaller node id to both
        min_node_id = min(gp1.node_id, gp2.node_id)

        # also convert other GPs with same node id to min_node_id
        for gp in all_plugged_gps:
            if gp.node_id == gp1.node_id or gp.node_id == gp2.node_id:
                gp.node_id = min_node_id

    return rest_results


def generate_netlist(rest_results: list[ResistorResult]) -> str:
    """
    Build and return netlist content (LTSpice format) as a string instead of writing to disk.
    """
    global top_power_supply_node_ids,bottom_power_supply_node_ids
    lines = []
    lines.append("* This is an autogenerated netlist file")
    lines.append(f"V1 N{top_power_supply_node_ids[0]} N{top_power_supply_node_ids[1]} V")
    lines.append(f"V2 N{bottom_power_supply_node_ids[0]} N{bottom_power_supply_node_ids[1]} V")

    for i, res in enumerate(rest_results):
        n1 = res.plugged_gps[0].node_id
        n2 = res.plugged_gps[1].node_id
        r = res.resistance
        
        # fallbacks for missing values
        n1_str = f"N{n1}" if n1 is not None else "N0"
        n2_str = f"N{n2}" if n2 is not None else "N0"

        lines.append(f"R{i} {n1_str} {n2_str} {r}")

    lines.append(".backanno")
    lines.append(".end")
    return "\n".join(lines) + "\n"
