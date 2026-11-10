## Loss-function ablation (U-Net, only the loss changes)

### DRIVE

| loss | best epoch | val dice | test dice | test iou | test acc | test sens | test spec |
|---|---|---|---|---|---|---|---|
| bce | 96 | 0.7876 | 0.7905 | 0.6536 | 0.9456 | 0.7616 | 0.9742 |
| focal | 90 | 0.7844 | 0.7901 | 0.6530 | 0.9411 | 0.8225 | 0.9595 |
| wbce | 91 | 0.7347 | 0.7584 | 0.6109 | 0.9227 | 0.9004 | 0.9261 |

### PH2

| loss | best epoch | val dice | test dice | test iou | test acc | test sens | test spec |
|---|---|---|---|---|---|---|---|
| bce | 93 | 0.9582 | 0.9069 | 0.8297 | 0.9459 | 0.9195 | 0.9564 |
| focal | 78 | 0.9443 | 0.8814 | 0.7879 | 0.9306 | 0.8994 | 0.9431 |
| wbce | 79 | 0.9525 | 0.8924 | 0.8057 | 0.9390 | 0.8821 | 0.9619 |
