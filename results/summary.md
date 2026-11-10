### test

| experiment | dataset | model | loss | best epoch | dice | iou | acc | sens | spec |
|---|---|---|---|---|---|---|---|---|---|
| drive_cnn | drive | cnn | wbce | 77 | 0.7419 | 0.5898 | 0.9197 | 0.8557 | 0.9297 |
| drive_unet | drive | unet | wbce | 91 | 0.7584 | 0.6109 | 0.9227 | 0.9004 | 0.9261 |
| drive_unet_bce | drive | unet | bce | 96 | 0.7905 | 0.6536 | 0.9456 | 0.7616 | 0.9742 |
| drive_unet_focal | drive | unet | focal | 90 | 0.7901 | 0.6530 | 0.9411 | 0.8225 | 0.9595 |
| ph2_cnn | ph2 | cnn | wbce | 100 | 0.9012 | 0.8202 | 0.9415 | 0.9299 | 0.9462 |
| ph2_unet | ph2 | unet | wbce | 79 | 0.8924 | 0.8057 | 0.9390 | 0.8821 | 0.9619 |
| ph2_unet_bce | ph2 | unet | bce | 93 | 0.9069 | 0.8297 | 0.9459 | 0.9195 | 0.9564 |
| ph2_unet_focal | ph2 | unet | focal | 78 | 0.8814 | 0.7879 | 0.9306 | 0.8994 | 0.9431 |

### val

| experiment | dataset | model | loss | best epoch | dice | iou | acc | sens | spec |
|---|---|---|---|---|---|---|---|---|---|
| drive_cnn | drive | cnn | wbce | 77 | 0.7125 | 0.5533 | 0.9173 | 0.8665 | 0.9241 |
| drive_unet | drive | unet | wbce | 91 | 0.7347 | 0.5806 | 0.9221 | 0.9115 | 0.9236 |
| drive_unet_bce | drive | unet | bce | 96 | 0.7876 | 0.6496 | 0.9518 | 0.7563 | 0.9780 |
| drive_unet_focal | drive | unet | focal | 90 | 0.7844 | 0.6453 | 0.9467 | 0.8193 | 0.9638 |
| ph2_cnn | ph2 | cnn | wbce | 100 | 0.9459 | 0.8974 | 0.9611 | 0.9638 | 0.9596 |
| ph2_unet | ph2 | unet | wbce | 79 | 0.9525 | 0.9094 | 0.9666 | 0.9491 | 0.9762 |
| ph2_unet_bce | ph2 | unet | bce | 93 | 0.9582 | 0.9197 | 0.9703 | 0.9649 | 0.9732 |
| ph2_unet_focal | ph2 | unet | focal | 78 | 0.9443 | 0.8945 | 0.9600 | 0.9611 | 0.9593 |

### train

| experiment | dataset | model | loss | best epoch | dice | iou | acc | sens | spec |
|---|---|---|---|---|---|---|---|---|---|
| drive_cnn | drive | cnn | wbce | 77 | 0.7452 | 0.5939 | 0.9232 | 0.9011 | 0.9263 |
| drive_unet | drive | unet | wbce | 91 | 0.7443 | 0.5927 | 0.9211 | 0.9219 | 0.9209 |
| drive_unet_bce | drive | unet | bce | 96 | 0.7804 | 0.6399 | 0.9473 | 0.7509 | 0.9753 |
| drive_unet_focal | drive | unet | focal | 90 | 0.7826 | 0.6428 | 0.9419 | 0.8382 | 0.9567 |
| ph2_cnn | ph2 | cnn | wbce | 100 | 0.9488 | 0.9027 | 0.9661 | 0.9711 | 0.9637 |
| ph2_unet | ph2 | unet | wbce | 79 | 0.9444 | 0.8946 | 0.9637 | 0.9516 | 0.9695 |
| ph2_unet_bce | ph2 | unet | bce | 93 | 0.9524 | 0.9091 | 0.9687 | 0.9653 | 0.9704 |
| ph2_unet_focal | ph2 | unet | focal | 78 | 0.9362 | 0.8800 | 0.9581 | 0.9491 | 0.9624 |
