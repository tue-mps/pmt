# PMT Model Zoo - DINOv3

> FPS measured on NVIDIA H100 with torch.compile (max-autotune).

## Panoptic Segmentation

### COCO

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">PQ</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: PMT-S 640x640 -->
<tr><td align="left"><a href="configs/coco/panoptic/pmt_s_640.yaml">PMT-S</a></td>
<td align="center">640×640</td>
<td align="center">400</td>
<td align="center">45.3</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_pmt_small_640_dinov3/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: PMT-B 640x640 -->
<tr><td align="left"><a href="configs/coco/panoptic/pmt_b_640.yaml">PMT-B</a></td>
<td align="center">640×640</td>
<td align="center">262</td>
<td align="center">52.8</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_pmt_base_640_dinov3/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: PMT-L 640x640 -->
<tr><td align="left"><a href="configs/coco/panoptic/pmt_l_640.yaml">PMT-L</a></td>
<td align="center">640×640</td>
<td align="center">141</td>
<td align="center">56.1</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_pmt_large_640_dinov3/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: PMT-L 1280x1280 -->
<tr><td align="left"><a href="configs/coco/panoptic/pmt_l_1280.yaml">PMT-L</a></td>
<td align="center">1280×1280</td>
<td align="center">29</td>
<td align="center">58.1</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_pmt_large_1280_dinov3/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>  

## Instance Segmentation

### COCO

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">mAP</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: PMT-L 640x640 -->
<tr><td align="left"><a href="configs/coco/instance/pmt_l_640.yaml">PMT-L</a></td>
<td align="center">640×640</td>
<td align="center">141</td>
<td align="center">45.4</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_instance_pmt_large_640_dinov3/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: PMT-L 1280x1280 -->
<tr><td align="left"><a href="configs/coco/instance/pmt_l_1280.yaml">PMT-L</a></td>
<td align="center">1280×1280</td>
<td align="center">29</td>
<td align="center">49.0</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_instance_pmt_large_1280_dinov3/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

## Semantic Segmentation

### ADE20K

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: PMT-L 512x512 -->
<tr><td align="left"><a href="configs/ade20k/semantic/pmt_l.yaml">PMT-L</a></td>
<td align="center">512×512</td>
<td align="center">128</td>
<td align="center">58.5</td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_semantic_pmt_large_512_dinov3/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

---

**Important:** The provided model weights are for the decoder only. The backbone (DINOv3) weights are not included and must be loaded separately.
