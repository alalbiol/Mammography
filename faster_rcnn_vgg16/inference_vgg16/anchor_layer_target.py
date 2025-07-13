import torch
import torch.nn as nn
import numpy as np
import yaml

# Assuming cfg is a dictionary-like object accessible globally or passed
# For demonstration, a simple mock cfg
class Cfg:
    def __init__(self):
        self.TRAIN = self.TrainCfg()
        self.EPS = np.finfo(np.float32).eps

    class TrainCfg:
        def __init__(self):
            self.RPN_CLOBBER_POSITIVES = False
            self.RPN_NEGATIVE_OVERLAP = 0.3
            self.RPN_POSITIVE_OVERLAP = 0.7
            self.RPN_FG_FRACTION = 0.5
            self.RPN_BATCHSIZE = 256
            self.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
            self.RPN_POSITIVE_WEIGHT = -1.0 # -1.0 means uniform weighting

cfg = Cfg()

# Placeholder for generate_anchors, bbox_overlaps, bbox_transform
# In a real project, these would be separate modules, likely in C/CUDA for speed.

def generate_anchors(base_size=16, ratios=(0.5, 1, 2), scales=(8, 16, 32)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    w, h, x_ctr, y_ctr = _whctrs(base_anchor)

    ws = np.round(np.sqrt(w * h / ratios))
    hs = np.round(ws * ratios)
    ws = np.expand_dims(ws, axis=1)
    hs = np.expand_dims(hs, axis=1)
    
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    
    # Enumerate all shifts
    # For Faster R-CNN, scales are applied to the base anchor sizes directly
    # rather than scaling the image or feature map.
    scaled_anchors = []
    for scale in scales:
        _w = w * scale
        _h = h * scale
        scaled_anchors.append(_mkanchors(np.array([_w]), np.array([_h]), x_ctr, y_ctr))
    
    # Combine anchors generated from different scales and ratios
    final_anchors = []
    for ratio in ratios:
        w_ratio = w * np.sqrt(ratio)
        h_ratio = h / np.sqrt(ratio)
        for scale in scales:
            sw = w_ratio * scale
            sh = h_ratio * scale
            final_anchors.append(_mkanchors(np.array([sw]), np.array([sh]), x_ctr, y_ctr))

    return np.vstack(final_anchors)

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), populate an array of anchors (x1, y1, x2, y2).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

# This is a simplified version, the original uses Cython for speed.
# For PyTorch, you'd want a torch-native (or CUDA-accelerated) implementation.
def bbox_overlaps(boxes, query_boxes):
    """
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns: overlaps (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    overlaps = torch.zeros((N, K), dtype=torch.float32, device=boxes.device)

    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + 1) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + 1))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]) + 1)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]) + 1)
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua
    return overlaps

# This is a simplified version. The original uses numpy and potentially optimized C code.
# For PyTorch, ensure all operations are on torch tensors.
def bbox_transform(ex_rois, gt_rois):
    """
    Compute bounding box regression targets from ex_rois to gt_rois.
    ex_rois: (N, 4)
    gt_rois: (N, 4)
    Returns: (N, 4) targets
    """
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets

DEBUG = False

class AnchorTargetLayer(nn.Module):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, scales=(8, 16, 32), allowed_border=0):
        super(AnchorTargetLayer, self).__init__()
        self._feat_stride = feat_stride
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales))).float()
        self._num_anchors = self._anchors.shape[0]
        self._allowed_border = allowed_border

        if DEBUG:
            print('anchors:')
            print(self._anchors)
            print('anchor shapes:')
            print(np.hstack((
                (self._anchors[:, 2] - self._anchors[:, 0]).cpu().numpy(),
                (self._anchors[:, 3] - self._anchors[:, 1]).cpu().numpy(),
            )))
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

    def forward(self, rpn_cls_score, gt_boxes, im_info):
        """
        rpn_cls_score: (N, A*2, H, W) - Placeholder for feature map shape to get H, W
        gt_boxes: (N, M, 5) - Ground truth boxes (x1, y1, x2, y2, label)
        im_info: (N, 3) - Image information (height, width, scale)
        """
        # Ensure batch size is 1 as per original Caffe layer
        assert rpn_cls_score.shape[0] == 1, \
            'Only single item batches are supported'

        device = rpn_cls_score.device

        height, width = rpn_cls_score.shape[-2:]
        im_info = im_info[0, :] # Original Caffe takes first item

        if DEBUG:
            print('')
            print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
            print('scale: {}'.format(im_info[2]))
            print('height, width: ({}, {})'.format(height, width))
            print('rpn: gt_boxes.shape', gt_boxes.shape)
            print('rpn: gt_boxes', gt_boxes)

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = torch.arange(0, width, device=device) * self._feat_stride
        shift_y = torch.arange(0, height, device=device) * self._feat_stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij') # Ensure correct meshgrid order
        
        shifts = torch.stack((shift_x.ravel(), shift_y.ravel(),
                              shift_x.ravel(), shift_y.ravel()), dim=1)

        A = self._num_anchors
        K = shifts.shape[0]
        
        # Original was numpy operations. In PyTorch, we can broadcast more directly.
        # all_anchors = (self._anchors.reshape((1, A, 4)) +
        #                shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        # all_anchors = all_anchors.reshape((K * A, 4))
        
        # Replicate anchors for each shift, and shifts for each anchor
        # Shape: (K, A, 4)
        all_anchors = (self._anchors.to(device).view(1, A, 4) +
                       shifts.view(K, 1, 4))
        all_anchors = all_anchors.view(K * A, 4)
        total_anchors = int(K * A)

        # only keep anchors inside the image
        inds_inside = torch.where(
            (all_anchors[:, 0] >= -self._allowed_border) &
            (all_anchors[:, 1] >= -self._allowed_border) &
            (all_anchors[:, 2] < im_info[1] + self._allowed_border) &  # width
            (all_anchors[:, 3] < im_info[0] + self._allowed_border)    # height
        )[0]

        if DEBUG:
            print('total_anchors', total_anchors)
            print('inds_inside', len(inds_inside))

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        if DEBUG:
            print('anchors.shape', anchors.shape)

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = torch.full((len(inds_inside), ), -1, dtype=torch.float32, device=device)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        # Note: bbox_overlaps expects numpy arrays in the original Caffe code,
        # so we convert back and forth for this placeholder. In a real PyTorch
        # implementation, bbox_overlaps would be a torch-native function.
        if gt_boxes.shape[1] == 0: # Handle cases with no ground truth boxes
            argmax_overlaps = torch.zeros(len(inds_inside), dtype=torch.int64, device=device)
            max_overlaps = torch.zeros(len(inds_inside), dtype=torch.float32, device=device)
            gt_argmax_overlaps = torch.zeros(0, dtype=torch.int64, device=device)
            gt_max_overlaps = torch.zeros(0, dtype=torch.float32, device=device)
            # No GT boxes means all anchors are negative (or don't care initially)
            labels.fill_(0) # Fill with 0 (negative) since there are no positives
        else:
            overlaps = bbox_overlaps(anchors, gt_boxes[:, :4]) # gt_boxes[:,:4] because last dim is label
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps[torch.arange(len(inds_inside), device=device), argmax_overlaps]
            gt_argmax_overlaps = overlaps.argmax(axis=0)
            gt_max_overlaps = overlaps[gt_argmax_overlaps, torch.arange(overlaps.shape[1], device=device)]
            # Find all anchors that are the best for *any* GT box (handling multiple GTs being best for one anchor)
            # This is tricky in PyTorch with direct comparison. Original Caffe used np.where(overlaps == gt_max_overlaps)[0]
            # which correctly gets indices of all occurrences.
            # For simplicity and common practice, we often just use gt_argmax_overlaps, which is sufficient if
            # an anchor can only be assigned to one GT (the one it overlaps most with).
            # The original logic `np.where(overlaps == gt_max_overlaps)[0]` implies that multiple anchors
            # might have the same highest overlap with *different* GTs, or that multiple anchors might tie
            # for highest overlap with a *single* GT.
            # A common alternative for `gt_argmax_overlaps` is to set `labels[max_overlaps == gt_max_overlaps[argmax_overlaps]] = 1`
            # or to iterate through GTs and find the best anchor for each.
            # For now, let's stick closer to the Caffe logic of finding *all* anchors that achieve `gt_max_overlaps`.
            # This is effectively finding anchors that are "best" for at least one GT.
            # A more robust way to implement `gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]`
            # in PyTorch would be:
            _, gt_max_overlaps_for_all_anchors = overlaps.max(dim=1)
            gt_argmax_overlaps_pyt = torch.nonzero(overlaps == gt_max_overlaps.unsqueeze(0), as_tuple=True)[0]

            if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels first so that positive labels can clobber them
                labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

            # fg label: for each gt, anchor with highest overlap
            labels[gt_argmax_overlaps_pyt] = 1 # Using the PyTorch equivalent for gt_argmax_overlaps

            # fg label: above threshold IOU
            labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

            if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels last so that negative labels can clobber positives
                labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = torch.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = fg_inds[torch.randperm(len(fg_inds), device=device)][:len(fg_inds) - num_fg]
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum(labels == 1).item() # Use .item() for scalar
        bg_inds = torch.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = bg_inds[torch.randperm(len(bg_inds), device=device)][:len(bg_inds) - num_bg]
            labels[disable_inds] = -1

        bbox_targets = torch.zeros((len(inds_inside), 4), dtype=torch.float32, device=device)
        if gt_boxes.shape[1] > 0: # Only compute targets if there are GT boxes
            # bbox_transform expects (N, 4) for both. gt_boxes[:, :4] extracts the coordinates.
            bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :4])

        bbox_inside_weights = torch.zeros((len(inds_inside), 4), dtype=torch.float32, device=device)
        bbox_inside_weights[labels == 1, :] = torch.tensor(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS, device=device)

        bbox_outside_weights = torch.zeros((len(inds_inside), 4), dtype=torch.float32, device=device)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = torch.sum(labels >= 0).item()
            if num_examples > 0: # Avoid division by zero
                positive_weights = torch.ones((1, 4), device=device) * 1.0 / num_examples
                negative_weights = torch.ones((1, 4), device=device) * 1.0 / num_examples
            else:
                positive_weights = torch.zeros((1, 4), device=device)
                negative_weights = torch.zeros((1, 4), device=device)
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            num_pos = torch.sum(labels == 1).item()
            num_neg = torch.sum(labels == 0).item()
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT / num_pos) if num_pos > 0 else 0
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) / num_neg) if num_neg > 0 else 0
            positive_weights = torch.full((1, 4), float(positive_weights), device=device)
            negative_weights = torch.full((1, 4), float(negative_weights), device=device)

        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        if DEBUG:
            # Convert to numpy for printing debug info
            self._sums += bbox_targets[labels == 1, :].sum(axis=0).cpu().numpy()
            self._squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0).cpu().numpy()
            self._counts += torch.sum(labels == 1).item()
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print('means:')
            print(means)
            print('stdevs:')
            print(stds)

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1, device=device)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0, device=device)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0, device=device)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0, device=device)

        if DEBUG:
            print('rpn: max max_overlap', torch.max(max_overlaps).item() if max_overlaps.numel() > 0 else -1)
            print('rpn: num_positive', torch.sum(labels == 1).item())
            print('rpn: num_negative', torch.sum(labels == 0).item())
            self._fg_sum += torch.sum(labels == 1).item()
            self._bg_sum += torch.sum(labels == 0).item()
            self._count += 1
            print('rpn: num_positive avg', self._fg_sum / self._count)
            print('rpn: num_negative avg', self._bg_sum / self._count)

        # labels
        # (1, A*height, width) -> (1, A, height, width) -> (1, 1, A*height, width)
        labels = labels.reshape((1, height, width, A)).permute(0, 3, 1, 2).contiguous()
        labels = labels.view(1, 1, A * height, width)

        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, A * 4)).permute(0, 3, 1, 2).contiguous()

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * 4)).permute(0, 3, 1, 2).contiguous()

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * 4)).permute(0, 3, 1, 2).contiguous()

        return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def _unmap(data, count, inds, fill=0, device='cpu'):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = torch.full((count, ), fill, dtype=torch.float32, device=device)
        ret[inds] = data
    else:
        ret = torch.full((count, ) + data.shape[1:], fill, dtype=torch.float32, device=device)
        ret[inds, :] = data
    return ret

# Helper functions for _compute_targets (re-using the ones defined earlier, ensuring they work with torch)
# These are already defined above, but ensure they are compatible with torch tensors.
# For example, `bbox_overlaps` and `bbox_transform` were initially written to accept numpy
# arrays and return numpy arrays. For full PyTorch compatibility, they should be converted
# to accept and return torch tensors directly, and use torch operations.
# The placeholder implementations provided for `bbox_overlaps` and `bbox_transform`
# *do* operate on PyTorch tensors, which is good.