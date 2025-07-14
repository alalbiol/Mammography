import torch
import torch.nn as nn
import numpy as np
import yaml


from generate_anchors import generate_anchors

DEBUG = False





def bbox_transform_inv(boxes, deltas):
    """
    Applies bounding-box transformations to the given boxes.
    Boxes are (x1, y1, x2, y2)
    Deltas are (dx, dy, dw, dh)
    """
    boxes = boxes.to(deltas.device) # Ensure boxes are on the same device as deltas

    if boxes.shape[0] == 0:
        return torch.zeros((0, deltas.shape[1]), dtype=deltas.dtype, device=deltas.device)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)

    pred_boxes = torch.zeros_like(deltas)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_info):
    """
    Clip boxes to image boundaries.
    im_info is (height, width)
    """
    height, width = im_info[0] / im_info[2], im_info[1] / im_info[2]  # Adjust for scale
    boxes[:, 0::4].clamp_(0, width - 1)
    boxes[:, 1::4].clamp_(0, height - 1)
    boxes[:, 2::4].clamp_(0, width - 1)
    boxes[:, 3::4].clamp_(0, height - 1)
    return boxes

# NMS wrapper (using torchvision's nms for efficiency)
try:
    from torchvision.ops import nms
except ImportError:
    print("torchvision not found. Please install it for NMS.")
    # Fallback for NMS if torchvision is not installed (basic CPU implementation)
    def nms(boxes, scores, iou_threshold):
        """
        Pure Python NMS.
        boxes: (N, 4) tensor
        scores: (N,) tensor
        iou_threshold: float
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort(descending=True)

        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i) # Appending a tensor, not a Python number

            if order.numel() == 1:
                break

            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])

            w = torch.max(torch.tensor(0.0, device=boxes.device), xx2 - xx1 + 1)
            h = torch.max(torch.tensor(0.0, device=boxes.device), yy2 - yy1 + 1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = torch.where(ovr <= iou_threshold)[0]
            order = order[inds + 1] # +1 because inds are relative to order[1:]
        
        if not keep:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        return torch.stack(keep)


class ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """
    def __init__(self, feat_stride, 
                anchor_scales=(8, 16, 32),
                anchor_ratios=[0.5, 1, 2],
                pre_nms_topN=12000,
                post_nms_topN=2000,
                nms_thresh=0.7,
                min_size=16):
        super(ProposalLayer, self).__init__()
        self._feat_stride = feat_stride
        self._anchors = generate_anchors(scales=torch.tensor(anchor_scales, dtype=torch.float32),
                                        ratios=torch.tensor(anchor_ratios, dtype=torch.float32))
        self._num_anchors = self._anchors.shape[0]
        self._pre_nms_topN = pre_nms_topN
        self._post_nms_topN = post_nms_topN
        self._nms_thresh = nms_thresh
        self._min_size = min_size

        if DEBUG:
            print('feat_stride: {}'.format(self._feat_stride))
            print('anchors:')
            print(self._anchors)

    def forward(self, scores, bbox_deltas, im_info):
        """
        Args:
            scores (torch.Tensor): RPN scores (1, 2*A, H, W) where A is num_anchors
                                The first A channels are background, second A are foreground.
            bbox_deltas (torch.Tensor): RPN bbox regression deltas (1, 4*A, H, W)
            im_info (torch.Tensor): Image information (1, 3) -> (height, width, scale)
        Returns:
            rois (torch.Tensor): Generated RoIs (R, 5) where each is (batch_idx, x1, y1, x2, y2)
            scores (torch.Tensor, optional): Scores for the RoIs (R, 1)
        """
        assert scores.shape[0] == 1, 'Only single item batches are supported'

        # Determine if in training or testing mode
        #cfg_key = 'TRAIN' if self.training else 'TEST'
        pre_nms_topN  = self._pre_nms_topN
        post_nms_topN = self._post_nms_topN
        nms_thresh = self._nms_thresh
        min_size = self._min_size

        # The first set of _num_anchors channels are bg probs, the second set are the fg probs
        # We need the foreground probabilities.
        # Original Caffe was `scores = bottom[0].data[:, self._num_anchors:, :, :]`
        # PyTorch equivalent: slice along dimension 1 (channel dimension)
        scores = scores[:, self._num_anchors:, :, :].contiguous() # Ensure contiguous memory

        im_height, im_width, im_scale = im_info[0, :3].cpu().numpy()

        if DEBUG:
            print('im_size: ({}, {})'.format(im_height, im_width))
            print('scale: {}'.format(im_scale))

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        if DEBUG:
            print('score map size: {}'.format(scores.shape))

        # Enumerate all shifts
        shift_x = torch.arange(0, width, device=scores.device) * self._feat_stride
        shift_y = torch.arange(0, height, device=scores.device) * self._feat_stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij') # Ensure correct order for meshgrid output
        
        # Original: np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
        # PyTorch equivalent:
        shifts = torch.stack((shift_x.reshape(-1), shift_y.reshape(-1),
                            shift_x.reshape(-1), shift_y.reshape(-1)), dim=1)

        # Enumerate all shifted anchors:
        A = self._num_anchors
        K = shifts.shape[0] # Number of spatial locations (H*W)
        
        # Original: self._anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        # This is equivalent to broadcasting the anchors to each shift location.
        # _anchors: (A, 4)
        # shifts: (K, 4)
        # We want (K, A, 4) by adding (1, A, 4) to (K, 1, 4)
        anchors = self._anchors.to(scores.device).reshape((1, A, 4)) + \
                shifts.reshape((K, 1, 4))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations
        # bbox_deltas: (1, 4 * A, H, W)
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4)
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).reshape(-1, 4)

        # Same for scores
        # scores: (1, A, H, W)
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1)
        scores = scores.permute(0, 2, 3, 1).reshape(-1, 1)

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to scaled image size (for this reason im_scale=1.0)
        proposals = clip_boxes(proposals, torch.tensor([im_height, im_width,1.0], device=proposals.device))

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, min_size * im_scale)
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN
        # `scores.ravel().argsort()[::-1]` is `scores.flatten().argsort(descending=True)`
        order = scores.flatten().argsort(descending=True)
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms
        # 7. take after_nms_topN
        # NMS expects (boxes, scores)
        
        proposals_pre_nms = proposals.cpu().numpy()
        scores_pre_nms = scores.cpu().numpy()
        
        
        keep = nms(proposals, scores.squeeze(1), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
            
        
        
        proposals = proposals[keep, :]
        scores = scores[keep] 
        nms_keep = keep.cpu().numpy() if keep.is_cuda else keep.numpy()   
        proposals_post_nms = proposals.cpu().numpy()
        scores_post_nms = scores.cpu().numpy()
        
        np.save('proposals_pre_nms.npy', proposals_pre_nms)
        np.save('scores_pre_nms.npy', scores_pre_nms)
        np.save('proposals_post_nms.npy', proposals_post_nms)
        np.save('scores_post_nms.npy', scores_post_nms)
        np.save('nms_keep.npy', nms_keep)
        
        

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = torch.zeros((proposals.shape[0], 1), dtype=torch.float32, device=proposals.device)
        rois = torch.cat((batch_inds, proposals), dim=1)

        # The original Caffe layer had an optional second output for scores.
        # In PyTorch, we can return multiple tensors.
        return rois, scores

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = torch.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

if __name__ == '__main__':
    # Example Usage:
    # Dummy inputs for demonstration
    batch_size = 1
    num_anchors = 9 # Based on default scales (8,16,32) and ratios (0.5,1,2) -> 3*3=9
    H, W = 38, 62 # Example feature map size

    # Simulate RPN output scores (foreground/background)
    # Caffe's scores shape: (N, 2*A, H, W)
    dummy_scores = torch.randn(batch_size, 2 * num_anchors, H, W)
    
    # Simulate RPN bbox deltas
    # Caffe's bbox_deltas shape: (N, 4*A, H, W)
    dummy_bbox_deltas = torch.randn(batch_size, 4 * num_anchors, H, W) * 0.1 # Small deltas

    # Simulate image info (height, width, scale)
    # Caffe's im_info shape: (N, 3)
    dummy_im_info = torch.tensor([[600., 1000., 1.6]], dtype=torch.float32) # Example: 600x1000 image, scale 1.6

    # Move tensors to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_scores = dummy_scores.to(device)
    dummy_bbox_deltas = dummy_bbox_deltas.to(device)
    dummy_im_info = dummy_im_info.to(device)

    # Initialize the ProposalLayer
    feat_stride = 16 # Common feature stride for VGG/ResNet
    proposal_layer = ProposalLayer(feat_stride=feat_stride).to(device)

    # Set to evaluation mode for testing (influences pre_nms_topN, post_nms_topN)
    proposal_layer.eval()

    # Perform forward pass
    print("Running ProposalLayer in eval mode...")
    with torch.no_grad():
        rois, roi_scores = proposal_layer(dummy_scores, dummy_bbox_deltas, dummy_im_info)

    print("\nGenerated RoIs shape:", rois.shape)
    print("Generated RoIs (first 5):\n", rois[:5])
    print("Generated RoI Scores shape:", roi_scores.shape)
    print("Generated RoI Scores (first 5):\n", roi_scores[:5])

    # Test in train mode
    proposal_layer.train()
    print("\nRunning ProposalLayer in train mode...")
    with torch.no_grad(): # Typically no_grad for inference, but demonstrating train mode settings
        rois_train, roi_scores_train = proposal_layer(dummy_scores, dummy_bbox_deltas, dummy_im_info)
    print("Generated RoIs shape (train mode):", rois_train.shape)
    print("Generated RoI Scores shape (train mode):", roi_scores_train.shape)

    # Verify that the outputs are different due to different NMS settings
    assert rois.shape[0] != rois_train.shape[0], "Expected different number of RoIs in train/eval mode due to NMS settings"
    print("\nTrain and Eval mode produced different numbers of RoIs as expected (due to NMS config).")