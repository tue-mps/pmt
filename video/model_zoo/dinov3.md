# PMT Model Zoo - DINOv3

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
<td align="center">69.2</td>
<td align="center">74.6</td>
<td align="center">113</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov3_segmenter/yt-2019_dinov3_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov3_online/yt-2019_dinov3_online_69.2.pth?download=true">Model Weights</a></td>
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
<td align="center">64.3</td>
<td align="center">69.0</td>
<td align="center">113</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov3_segmenter/yt-2021_dinov3_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov3_online/yt-2021_dinov3_online_64.3.pth?download=true">Model Weights</a></td>
</tr>
</tbody></table>

### OVIS

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
<tr><td align="left"><a href="../configs/ovis/videomt/vit-large/videomt_online_ViTL.yaml">PMT-L</a></td>
<td align="center">52.0</td>
<td align="center">57.7</td>
<td align="center">97</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov3_segmenter/ovis_dinov3_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov3_online/ovis_dinov3_online_52.0.pth?download=true">Model Weights</a></td>
</tr>
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
<td align="center">55.5</td>
<td align="center">49.2</td>
<td align="center">58</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov3_segmenter/vipseg_dinov3_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov3_online/vipseg_dinov3_online_55.5_49.2.pth?download=true">Model Weights</a></td>
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
<td align="center">94.9</td>
<td align="center">65.7</td>
<td align="center">57</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov3_segmenter/vspw_dinov3_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov3_online/vspw_dinov3_online_94.9_65.7.pth?download=true">Model Weights</a></td>
</tr>
</tbody></table>
