# PMT Model Zoo - DINOv2

> FPS measured on NVIDIA H100 with default torch.compile.

## Video Instance Segmentation

### YouTube-VIS 2019

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">AP</th>
<th valign="bottom">AR<sub>10</sub></th>
<th valign="bottom">FPS</th>
<th valign="bottom">Init Weights</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: PMT-L 640x640 -->
<tr><td align="left"><a href="../configs/ytvis19/videomt/vit-large/videomt_online_ViTL.yaml">PMT-L</a></td>
<td align="center">68.8</td>
<td align="center">73.9</td>
<td align="center">124</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov2_segmenter/yt-2019_dinov2_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov2_online/yt-2019_dinov2_online_68.8.pth?download=true">Model Weights</a></td>
</tr>
</tbody></table>  




### YouTube-VIS 2021

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">AP</th>
<th valign="bottom">AR<sub>10</sub></th>
<th valign="bottom">FPS</th>
<th valign="bottom">Init Weights</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<tr><td align="left"><a href="../configs/ytvis21/videomt/vit-large/videomt_online_ViTL.yaml">PMT-L</a></td>
<td align="center">63.8</td>
<td align="center">68.1</td>
<td align="center">124</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov2_segmenter/yt-2021_dinov2_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov2_online/yt-2021_dinov2_online_63.8.pth?download=true">Model Weights</a></td>

</tbody></table>

## Video Panoptic Segmentation

### VIPSeg

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">VPQ</th>
<th valign="bottom">STQ</th>
<th valign="bottom">FPS</th>
<th valign="bottom">Init Weights</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: PMT-L 1024x1024 -->
<tr><td align="left"><a href="../configs/VIPSeg/videomt/vit-large/videomt_Online_ViTL.yaml">PMT-L</a></td>
<td align="center">55.3</td>
<td align="center">48.2</td>
<td align="center">60</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov2_segmenter/vipseg_dinov2_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov2_online/vipseg_dinov2_online_55.3_48.2.pth?download=true">Model Weights</a></td>
</tr>
</tbody></table>

## Video Semantic Segmentation
### VSPW

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">mVC<sub>16</sub></th>
<th valign="bottom">mIoU</th>
<th valign="bottom">FPS</th>
<th valign="bottom">Init Weights</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: PMT-L 512x512 -->
<tr><td align="left"><a href="../configs/VSPW/videomt/vit-large/videomt_online_ViTL.yaml">PMT-L</a></td>
<td align="center">94.6</td>
<td align="center">64.3</td>
<td align="center">60</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov2_segmenter/vspw_dinov2_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov2_online/vspw_dinov2_online_94.6_64.3.pth?download=true">Model Weights</a></td>
</tr>
</tbody></table>

