import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from collections import OrderedDict
import torchvision

from proposal_layer import ProposalLayer, bbox_transform_inv, clip_boxes

# Import necessary for ROI Align and NMS (install torchvision if you haven't)
try:
    from torchvision.ops import roi_pool, roi_align, nms
except ImportError:
    print("16 torchvision not found. Please install it for ROI Pooling/Align/NMS support.")
    roi_pool = None
    roi_align = None
    nms = None


DEBUG = False


class VGG16Backbone(nn.Module):
    def __init__(self):
        super(VGG16Backbone, self).__init__()
        # VGG-16 layers from the Caffe prototxt
        # Note: In Caffe, ReLU layers are often in-place.
        # Here we define them explicitly after convolution.

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # Note: pool5 is not explicitly defined here as it's typically handled
        # by the ROI Pooling/Align layer, taking 'conv5_3' as input.

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        return x

class RPN(nn.Module):
    def __init__(self, in_channels, num_anchors=9):
        super(RPN, self).__init__()
        self.rpn_conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.rpn_cls_score = nn.Conv2d(512, num_anchors * 2, kernel_size=1, padding=0) # 2 (bg/fg) * num_anchors
        self.rpn_bbox_pred = nn.Conv2d(512, num_anchors * 4, kernel_size=1, padding=0) # 4 (coords) * num_anchors

    def forward(self, x):
        rpn_out = F.relu(self.rpn_conv(x))
        rpn_cls_score = self.rpn_cls_score(rpn_out)
        rpn_bbox_pred = self.rpn_bbox_pred(rpn_out)
        return rpn_cls_score, rpn_bbox_pred

class RCNNHead(nn.Module):
    def __init__(self, num_classes):
        super(RCNNHead, self).__init__()
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.cls_score = nn.Linear(4096, num_classes) # num_classes (including background)
        self.bbox_pred = nn.Linear(4096, num_classes * 4) # 4 coords * num_classes

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the pooled features
        x = F.relu(self.fc6(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, p=0.5, training=self.training)
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_score, bbox_pred

class FasterRCNN(nn.Module):
    def __init__(self, num_classes, use_roi_align=True):
        super(FasterRCNN, self).__init__()
        self.num_classes = num_classes
        self.use_roi_align = use_roi_align
        self.feat_stride = 16 # VGG-16 backbone reduces spatial dimensions by 16 (2*2*2*2)
        self.pooled_h = 7
        self.pooled_w = 7
        self.spatial_scale = 1.0 / self.feat_stride # 0.0625

        self.backbone = VGG16Backbone()
        self.rpn = RPN(in_channels=512) # conv5_3 has 512 output channels
        self.rcnn_head = RCNNHead(num_classes)

        if not roi_pool or not roi_align:
            print("Warning: torchvision ops (roi_pool, roi_align) not available. Please install torchvision for ROI operations.")

                # Initialize the custom AnchorGenerator


        # Initialize the ProposalLayer for inference
        # These are common default values, you might need to tune them
        self.proposal_layer = ProposalLayer(
            feat_stride=self.feat_stride,
            anchor_scales=(8, 16, 32), # Scales for anchors
            anchor_ratios=[0.5, 1, 2], # Aspect ratios for anchors
            pre_nms_topN=6000, # Number of proposals before NMS
            post_nms_topN=300, # Number of proposals after NMS for inference
            nms_thresh=0.7,
            min_size=self.feat_stride # Minimum proposal size, often related to feature stride
        )


    def forward(self, images, im_info, gt_boxes=None, rois=None):
        # images: (N, C, H, W)
        # im_info: (N, 3) - [height, width, scale]
        # gt_boxes: (G, 5) - [x1, y1, x2, y2, class_id] - only for training
        # rois: (R, 5) - [batch_idx, x1, y1, x2, y2] - for inference, or from RPN during training

        assert im_info.shape[0] == images.shape[0], "Batch size mismatch between images and im_info"
        assert im_info.shape[0] == 1, "Currently only batch size of 1 is supported for inference"

        base_feat = self.backbone(images)

        # RPN branch
        rpn_cls_score, rpn_bbox_pred = self.rpn(base_feat)
        
        rpn_cls_score_reshape = rpn_cls_score.reshape(1, 2, -1, rpn_cls_score.shape[-1])
        rpn_cls_prob = torch.softmax(rpn_cls_score_reshape, axis=1)
        rpn_cls_prob_reshape = rpn_cls_prob.reshape(1, rpn_cls_score.shape[1], -1, rpn_cls_score.shape[-1])
        
        
        
        # For batch processing, we iterate through images in the batch.
        # The ProposalLayer is designed to handle this.
        rois, scores = self.proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info)
        
        if DEBUG:
            print("Scores after proposal layer:", scores.shape)  # Debugging output
            print("First scores", scores[:10])  # Print first score for debugging
            print("First rois", rois[:10])  # Print first score for debugging

        if rois is None:
            # Placeholder for RPN's ProposalLayer equivalent during inference/eval
            # In a real scenario, this would use rpn_cls_score, rpn_bbox_pred, im_info
            # to generate proposals.
            # For demonstration, if no rois are provided, we'll return RPN outputs
            # and expect proposals to be handled externally or in a dedicated module.
            print("Warning: No ROIs provided. RCNN head will not be executed.")
            return rpn_cls_score, rpn_bbox_pred, None, None, None


        # ROI Pooling / ROI Align
        if self.use_roi_align and roi_align:
            # roi_align expects ROIs in (batch_index, x1, y1, x2, y2) format
            # and coordinates normalized to the feature map scale.
            # spatial_scale already handles this.
            pooled_features = roi_align(base_feat, rois,
                                        output_size=(self.pooled_h, self.pooled_w),
                                        spatial_scale=self.spatial_scale)
        elif roi_pool:
            pooled_features = roi_pool(base_feat, rois,
                                    output_size=(self.pooled_h, self.pooled_w),
                                    spatial_scale=self.spatial_scale)
            
# The above Python code snippet is a part of a neural network model implementation. It seems to be
# related to handling features at a specific layer of the network. Here is a breakdown of the code:
            # print("spatial scale:", self.spatial_scale)
            
            # print("Saving pooled features shape:", pooled_features.shape)  # Debugging output
            # pooled_features_np = pooled_features.cpu().numpy()
            # np.save("pool5.npy", pooled_features_np)
            
            # print("replacing pooled_features with pooled_features_caffe")
            # pooled_features_caffe = np.load("../scoring/P_00001_LEFT_CC_pool5.npy")
            # pooled_features = torch.from_numpy(pooled_features_caffe).to(pooled_features.device)

        
        else:
            raise RuntimeError("367 ROI Pooling/Align not available. Please install torchvision.")

        # R-CNN head
        cls_score, bbox_pred = self.rcnn_head(pooled_features)

        return rpn_cls_score, rpn_bbox_pred, cls_score, bbox_pred, rois

def load_caffe_weights_into_pytorch(pytorch_model, extracted_weights_dir):
    """
    Loads weights and biases from extracted .npy files into the PyTorch model.
    """
    print(f"\nLoading Caffe weights from '{extracted_weights_dir}' into PyTorch model...")

    # Mapping Caffe layer names to PyTorch module names and parameter names
    # This is crucial for correctly mapping the weights.
    # Caffe: {layer_name}_weights.npy, {layer_name}_biases.npy
    # PyTorch: module_name.weight, module_name.bias

    # VGG Backbone mapping
    caffe_to_pytorch_map = {
        'conv1_1': 'backbone.conv1_1', 'conv1_2': 'backbone.conv1_2',
        'conv2_1': 'backbone.conv2_1', 'conv2_2': 'backbone.conv2_2',
        'conv3_1': 'backbone.conv3_1', 'conv3_2': 'backbone.conv3_2', 'conv3_3': 'backbone.conv3_3',
        'conv4_1': 'backbone.conv4_1', 'conv4_2': 'backbone.conv4_2', 'conv4_3': 'backbone.conv4_3',
        'conv5_1': 'backbone.conv5_1', 'conv5_2': 'backbone.conv5_2', 'conv5_3': 'backbone.conv5_3',
        'rpn_conv/3x3': 'rpn.rpn_conv',
        'rpn_cls_score': 'rpn.rpn_cls_score',
        'rpn_bbox_pred': 'rpn.rpn_bbox_pred',
        'fc6': 'rcnn_head.fc6',
        'fc7': 'rcnn_head.fc7',
        'cls_score': 'rcnn_head.cls_score',
        'bbox_pred': 'rcnn_head.bbox_pred',
    }

    model_state_dict = pytorch_model.state_dict()
    print("Number of parameters in PyTorch model:", len(model_state_dict))

    num_imported_weights = 0
    for caffe_layer_name, pytorch_module_prefix in caffe_to_pytorch_map.items():
        # Load weights
        weights_path = os.path.join(extracted_weights_dir, f"{caffe_layer_name}_weights.npy")
        if os.path.exists(weights_path):
            caffe_weights = np.load(weights_path)
            # Caffe convolutional weights are (out_channels, in_channels, kH, kW)
            # PyTorch convolutional weights are (out_channels, in_channels, kH, kW) - direct match!
            # Caffe InnerProduct weights are (out_channels, in_channels)
            # PyTorch Linear weights are (out_features, in_features) - direct match!
            # Ensure the shapes match before loading
            try:
                if f"{pytorch_module_prefix}.weight" in model_state_dict:
                    # For convolutional layers
                    if len(caffe_weights.shape) == 4:
                        # Caffe weights (out, in, h, w)
                        # PyTorch weights (out, in, h, w)
                        expected_shape = model_state_dict[f"{pytorch_module_prefix}.weight"].shape
                        if caffe_weights.shape != expected_shape:
                            print(f"Warning: Weight shape mismatch for {caffe_layer_name}: Caffe {caffe_weights.shape} vs PyTorch {expected_shape}. Transposing if necessary.")
                            # Attempt to transpose if it's a common Caffe-to-PyTorch conv weight mismatch (unlikely for VGG)
                            # E.g., Caffe (out, in, kH, kW) -> PyTorch (out, in, kH, kW)
                            # If they don't match, you might need to investigate specific layer configurations.
                            # For VGG, it should generally be a direct match.
                        model_state_dict[f"{pytorch_module_prefix}.weight"].copy_(torch.from_numpy(caffe_weights))
                        num_imported_weights += 1
                        if DEBUG:
                            print(f"  Loaded {caffe_layer_name}_weights into {pytorch_module_prefix}.weight")

                    # For InnerProduct (Linear) layers
                    elif len(caffe_weights.shape) == 2:
                        # Caffe FC weights are (out_features, in_features)
                        # PyTorch Linear weights are (out_features, in_features)
                        # So, a direct copy is usually sufficient.
                        # However, for VGG, the fully connected layers typically
                        # connect from a flattened convolutional output.
                        # Caffe's InnerProduct can behave like a matrix multiplication
                        # where the input is flattened.
                        # PyTorch's Linear layer expects a 2D input (batch_size, in_features).
                        # The Caffe weights are often stored as (output_dim, input_dim).
                        # PyTorch's Linear.weight is (out_features, in_features).
                        # For VGG's fc6/fc7, Caffe's shape is usually (output_channels, input_channels).
                        # PyTorch's Linear expects (output_features, input_features).
                        # Often, Caffe stores FC weights as (out_dim, in_dim).
                        # PyTorch's `nn.Linear` `weight` is (out_features, in_features).
                        # So, we need to transpose Caffe's (in_dim, out_dim) to (out_dim, in_dim) if that's the case.
                        # Let's check the common VGG FC behavior.
                        # For VGG, Caffe's FC weights are usually (output_dim, input_dim).
                        # PyTorch's `nn.Linear` weights are also (output_dim, input_dim).
                        # So, direct copy for FC layers should be fine.
                        expected_shape = model_state_dict[f"{pytorch_module_prefix}.weight"].shape
                        if caffe_weights.shape != expected_shape:
                             # This is a common point of failure for Caffe FC to PyTorch Linear
                             # Caffe's InnerProduct might have its weights stored in a different order (e.g., (in_dim, out_dim))
                             # or might expect a flattened input in a different channel order.
                             # If a mismatch occurs, try transposing.
                             print(f"Warning: FC weight shape mismatch for {caffe_layer_name}: Caffe {caffe_weights.shape} vs PyTorch {expected_shape}. Attempting transpose.")
                             import sys
                             sys.exit()
                             try:
                                 model_state_dict[f"{pytorch_module_prefix}.weight"].copy_(torch.from_numpy(caffe_weights.T))
                                 print(f"  Loaded {caffe_layer_name}_weights into {pytorch_module_prefix}.weight (transposed)")
                                 num_imported_weights += 1
                             except Exception as transpose_e:
                                 print(f"  Failed to transpose and load: {transpose_e}")
                                 print(f"  Please manually verify weight dimensions for {caffe_layer_name}.")
                        else:
                            model_state_dict[f"{pytorch_module_prefix}.weight"].copy_(torch.from_numpy(caffe_weights))
                            if DEBUG:
                                print(f"  Loaded {caffe_layer_name}_weights into {pytorch_module_prefix}.weight")
                            num_imported_weights += 1


            except RuntimeError as e:
                print(f"Error loading weights for {caffe_layer_name} (into {pytorch_module_prefix}.weight): {e}")
                print(f"Caffe weights shape: {caffe_weights.shape}, PyTorch expected shape: {model_state_dict[f'{pytorch_module_prefix}.weight'].shape}")

        # Load biases
        biases_path = os.path.join(extracted_weights_dir, f"{caffe_layer_name}_biases.npy")
        if os.path.exists(biases_path):
            caffe_biases = np.load(biases_path)
            try:
                model_state_dict[f"{pytorch_module_prefix}.bias"].copy_(torch.from_numpy(caffe_biases))
                if DEBUG:
                    print(f"  Loaded {caffe_layer_name}_biases into {pytorch_module_prefix}.bias")
                num_imported_weights += 1
            except RuntimeError as e:
                print(f"Error loading biases for {caffe_layer_name} (into {pytorch_module_prefix}.bias): {e}")
                print(f"Caffe biases shape: {caffe_biases.shape}, PyTorch expected shape: {model_state_dict[f'{pytorch_module_prefix}.bias'].shape}")

    pytorch_model.load_state_dict(model_state_dict)
    print("All available Caffe weights loaded into the PyTorch model.")
    print(f"Total parameters imported: {num_imported_weights} out of {len(model_state_dict)}")

def decode_boxes(rois, bbox_pred, num_classes, image_sizes):
    """
    Decodes bounding box predictions from deltas to actual coordinates.

    Args:
        rois (Tensor): Shape (Total_Num_Proposals, 5) - (batch_idx, x1, y1, x2, y2)
        bbox_pred (Tensor): Shape (Total_Num_Proposals, num_classes * 4) - predicted deltas
        num_classes (int): Total number of classes (including background).
        image_sizes (list of tuples): List of (H, W) for each image in the batch.

    Returns:
        Tensor: Decoded bounding boxes (Total_Num_Proposals, num_classes, 4) in (x1, y1, x2, y2) format.
    """
    if rois.numel() == 0:
        return torch.empty(0, num_classes, 4, dtype=rois.dtype, device=rois.device)

    # Extract proposal boxes (x1, y1, x2, y2)
    proposals = rois[:, 1:] # (Total_Num_Proposals, 4)
    batch_indices = rois[:, 0].long() # (Total_Num_Proposals,)

    # Convert proposals from (x1, y1, x2, y2) to (cx, cy, w, h)
    widths = proposals[:, 2] - proposals[:, 0] + 1
    heights = proposals[:, 3] - proposals[:, 1] + 1
    ctr_x = proposals[:, 0] + 0.5 * widths
    ctr_y = proposals[:, 1] + 0.5 * heights

    # Reshape bbox_pred to (Total_Num_Proposals, num_classes, 4)
    bbox_pred = bbox_pred.view(-1, num_classes, 4)

    # Apply deltas for each class
    # dx, dy, dw, dh are (Total_Num_Proposals, num_classes)
    dx = bbox_pred[:, :, 0]
    dy = bbox_pred[:, :, 1]
    dw = bbox_pred[:, :, 2]
    dh = bbox_pred[:, :, 3]

    # Expand widths/heights/centers to match the num_classes dimension
    widths = widths.unsqueeze(1) # (Total_Num_Proposals, 1)
    heights = heights.unsqueeze(1) # (Total_Num_Proposals, 1)
    ctr_x = ctr_x.unsqueeze(1) # (Total_Num_Proposals, 1)
    ctr_y = ctr_y.unsqueeze(1) # (Total_Num_Proposals, 1)

    # Predicted centers and dimensions
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    # Convert back to (x1, y1, x2, y2)
    decoded_boxes = torch.stack([
        pred_ctr_x - 0.5 * pred_w,
        pred_ctr_y - 0.5 * pred_h,
        pred_ctr_x + 0.5 * pred_w,
        pred_ctr_y + 0.5 * pred_h
    ], dim=-1) # (Total_Num_Proposals, num_classes, 4)

    # Clip decoded boxes to image boundaries
    # Need to get image dimensions for each proposal's batch_idx
    img_widths = torch.tensor([img_size[1] for img_size in image_sizes], dtype=decoded_boxes.dtype, device=decoded_boxes.device)[batch_indices]
    img_heights = torch.tensor([img_size[0] for img_size in image_sizes], dtype=decoded_boxes.dtype, device=decoded_boxes.device)[batch_indices]

    # --- FIX STARTS HERE ---
    # Create a zero tensor for the 'min' argument, ensuring it's on the same device and dtype
    zero_tensor = torch.tensor(0.0, dtype=decoded_boxes.dtype, device=decoded_boxes.device)

    decoded_boxes[:, :, 0].clamp_(min=zero_tensor, max=img_widths.unsqueeze(1) - 1)
    decoded_boxes[:, :, 1].clamp_(min=zero_tensor, max=img_heights.unsqueeze(1) - 1)
    decoded_boxes[:, :, 2].clamp_(min=zero_tensor, max=img_widths.unsqueeze(1) - 1)
    decoded_boxes[:, :, 3].clamp_(min=zero_tensor, max=img_heights.unsqueeze(1) - 1)
    # --- FIX ENDS HERE ---

    return decoded_boxes

def post_process_detections(cls_score_out, bbox_pred_out, rois_out, im_info, num_classes, score_thresh=0.05, 
                            nms_thresh=0.3):
    """
    Performs final post-processing on the model outputs to get detected objects.

    Args:
        cls_score_out (Tensor): R-CNN classification scores (Total_Num_Proposals, num_classes).
        bbox_pred_out (Tensor): R-CNN bounding box predictions (Total_Num_Proposals, num_classes * 4).
        rois_out (Tensor): Generated ROIs from ProposalLayer (Total_Num_Proposals, 5).
        im_info (Tensor): Original image information (N, 3) - [height, width, scale].
        num_classes (int): Total number of classes (including background).
        score_thresh (float): Confidence threshold for filtering detections.
        nms_thresh (float): IoU threshold for Non-Maximum Suppression. Should be 0.3 to match Caffe model.

    Returns:
        List[Dict]: A list of dictionaries, one for each image in the batch.
                    Each dictionary contains 'boxes', 'labels', 'scores'.
    """
    if nms is None:
        raise ImportError("torchvision.ops.nms not found. Cannot perform post-processing without NMS.")

    batch_size = im_info.shape[0]
    device = cls_score_out.device
    
    # Get image sizes from im_info for clipping
    image_sizes = [(int(im_info[i, 0].item()), int(im_info[i, 1].item())) for i in range(batch_size)]

    # Decode bounding boxes
    # decoded_boxes will be (Total_Num_Proposals, num_classes, 4)
    decoded_boxes = decode_boxes(rois_out, bbox_pred_out, num_classes, image_sizes)

    # Apply softmax to get class probabilities
    class_probs = F.softmax(cls_score_out, dim=-1) # (Total_Num_Proposals, num_classes)

    final_detections = [[] for _ in range(batch_size)]

    # Iterate through each image in the batch
    for img_idx in range(batch_size):
        # Select proposals belonging to the current image
        img_rois_mask = (rois_out[:, 0].long() == img_idx)
        
        if not img_rois_mask.any():
            # No proposals for this image, continue to next
            continue

        img_class_probs = class_probs[img_rois_mask] # (Num_Proposals_for_Img, num_classes)
        img_decoded_boxes = decoded_boxes[img_rois_mask] # (Num_Proposals_for_Img, num_classes, 4)

        img_boxes = []
        img_labels = []
        img_scores = []

        # Iterate through each class (excluding background, class 0)
        for cls_id in range(1, num_classes): # Assuming class 0 is background
            cls_scores = img_class_probs[:, cls_id] # (Num_Proposals_for_Img,)
            cls_boxes = img_decoded_boxes[:, cls_id, :] # (Num_Proposals_for_Img, 4)

            # Filter by confidence threshold
            keep_score = cls_scores >= score_thresh
            cls_scores = cls_scores[keep_score]
            cls_boxes = cls_boxes[keep_score]

            if cls_scores.numel() == 0:
                continue

            # Apply NMS per class
            # Add a large offset to boxes for NMS to work across classes if needed,
            # but torchvision.ops.nms assumes per-class NMS is handled by iterating classes.
            # If you were to do NMS across all classes at once, you'd add class_id * large_offset to boxes.
            # However, standard Faster R-CNN applies NMS per class.
            keep_nms = nms(cls_boxes, cls_scores, nms_thresh)

            # Store final detections for this class
            img_boxes.append(cls_boxes[keep_nms])
            img_labels.append(torch.full((len(keep_nms),), cls_id, dtype=torch.long, device=device))
            img_scores.append(cls_scores[keep_nms])
        
        # Concatenate results for the current image
        if len(img_boxes) > 0:
            final_detections[img_idx] = {
                'boxes': torch.cat(img_boxes, dim=0),
                'labels': torch.cat(img_labels, dim=0),
                'scores': torch.cat(img_scores, dim=0)
            }
        else:
            # No detections for this image
            final_detections[img_idx] = {
                'boxes': torch.empty(0, 4, dtype=torch.float32, device=device),
                'labels': torch.empty(0, dtype=torch.long, device=device),
                'scores': torch.empty(0, dtype=torch.float32, device=device)
            }

    return final_detections


def post_process_detections2(cls_score_out, bbox_pred_out, rois, im_info, num_classes, score_thresh=0.05, 
                            nms_thresh=0.3):
    """
    Performs final post-processing on the model outputs to get detected objects.

    Args:
        cls_score_out (Tensor): R-CNN classification scores (Total_Num_Proposals, num_classes).
        bbox_pred_out (Tensor): R-CNN bounding box predictions (Total_Num_Proposals, num_classes * 4).
        rois_out (Tensor): Generated ROIs from ProposalLayer (Total_Num_Proposals, 5).
        im_info (Tensor): Original image information (N, 3) - [height, width, scale].
        num_classes (int): Total number of classes (including background).
        score_thresh (float): Confidence threshold for filtering detections.
        nms_thresh (float): IoU threshold for Non-Maximum Suppression. Should be 0.3 to match Caffe model.

    Returns:
        List[Dict]: A list of dictionaries, one for each image in the batch.
                    Each dictionary contains 'boxes', 'labels', 'scores'.
    """
    if nms is None:
        raise ImportError("torchvision.ops.nms not found. Cannot perform post-processing without NMS.")

    batch_size = im_info.shape[0]
    
    assert batch_size == 1, "Currently only batch size of 1 is supported for inference"
    device = cls_score_out.device
    
   
    # Apply softmax to get class probabilities
    class_probs = F.softmax(cls_score_out, dim=-1) # (Total_Num_Proposals, num_classes)

    ind = torch.argmax(class_probs[:,2]) # (Total_Num_Proposals,)
    

    all_boxes = [[]  for _ in range(num_classes)]# Initialize a list for each class)]
                

    # Iterate through each image in the batch
    for img_idx in range(batch_size):
        boxes = rois[:, 1:5] / im_info[0,2] # Scale ROIs to original image size
        pred_boxes = bbox_transform_inv(boxes, bbox_pred_out)
        pred_boxes = clip_boxes(pred_boxes, im_info[0,:])
        
    
        # skip j = 0, because it's the background class
        for j in range(1, num_classes):
            inds = torch.nonzero(class_probs[:, j] > score_thresh, as_tuple=False).squeeze(1)
            if len(inds) == 0:
                continue
            cls_scores = class_probs[inds, j]
            cls_boxes = pred_boxes[inds, j*4:(j+1)*4]
            keep_nms = nms(cls_boxes, cls_scores, nms_thresh)
            
            cls_boxes = cls_boxes.cpu().numpy()
            cls_scores = cls_scores.cpu().numpy()
            keep_nms = keep_nms.cpu().numpy()

            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)

            cls_dets = cls_dets[keep_nms, :]
            all_boxes[j] = cls_dets

    return all_boxes



# --- Example Usage ---
if __name__ == "__main__":
    # 1. Define paths (replace with your actual paths)
    EXTRACTED_WEIGHTS_DIR = "../faster_rcnn_vgg16_weights" # Output directory from your extraction script

    # 2. Instantiate the PyTorch model
    NUM_CLASSES = 3 # From your Caffe prototxt: 'num_output: 3' for cls_score, and 'num_classes': 3 for roi-data

    # Choose between ROI Align and ROI Pooling
    # Set to True for ROI Align, False for ROI Pooling
    use_roi_align_option = False
    pytorch_model = FasterRCNN(num_classes=NUM_CLASSES, use_roi_align=use_roi_align_option)
    pytorch_model.eval() # Set to evaluation mode for inference

    # 3. Load the extracted Caffe weights
    if os.path.exists(EXTRACTED_WEIGHTS_DIR):
        load_caffe_weights_into_pytorch(pytorch_model, EXTRACTED_WEIGHTS_DIR)
    else:
        print(f"Directory '{EXTRACTED_WEIGHTS_DIR}' not found. Cannot load extracted weights.")
        print("Please run the Caffe weight extraction script first.")

    # 4. Dummy input for testing forward pass
    # These are placeholders and need to be adapted to your actual data pipeline.
    dummy_image = torch.randn(1, 3, 600, 800) # Example image size
    dummy_im_info = torch.tensor([[600., 800., 1.0]]) # height, width, scale
    # Example ROIs: (batch_idx, x1, y1, x2, y2)
    # These would typically come from the RPN ProposalLayer in a full Faster R-CNN.
    dummy_rois = torch.tensor([
        [0, 10, 20, 100, 150],
        [0, 50, 60, 200, 250],
        [0, 300, 350, 500, 550]
    ], dtype=torch.float32)

    print(f"\n--- Testing PyTorch model forward pass (using ROI {'Align' if use_roi_align_option else 'Pooling'}) ---")
    with torch.no_grad():
        rpn_cls_score_out, rpn_bbox_pred_out, cls_score_out, bbox_pred_out, rois_out = \
            pytorch_model(dummy_image, dummy_im_info)

    print(f"RPN Classification Score output shape: {rpn_cls_score_out.shape}")
    print(f"RPN Bounding Box Prediction output shape: {rpn_bbox_pred_out.shape}")
    if cls_score_out is not None:
        print(f"RCNN Classification Score output shape: {cls_score_out.shape}")
        print(f"RCNN Bounding Box Prediction output shape: {bbox_pred_out.shape}")

    print("\n--- Performing Post-processing and Final NMS ---")
    # Pass the original im_info (which contains height and width for each image in the batch)
    final_detections_per_image = post_process_detections(
        cls_score_out, bbox_pred_out, rois_out, dummy_im_info,
        num_classes=NUM_CLASSES, score_thresh=0.05, nms_thresh=0.9
    )

    for i, detections in enumerate(final_detections_per_image):
        print(f"\n--- Detections for Image {i+1} ---")
        if len(detections['boxes']) > 0:
            for j in range(len(detections['boxes'])):
                box = detections['boxes'][j].tolist()
                label = detections['labels'][j].item()
                score = detections['scores'][j].item()
                print(f"  Box: [{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f}], Label: {label}, Score: {score:.4f}")
        else:
            print("  No detections found for this image.")