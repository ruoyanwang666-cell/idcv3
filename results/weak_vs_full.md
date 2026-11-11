## Point supervision vs full supervision (PH2 test set, U-Net, cross entropy)

Full supervision (all pixels labelled): test Dice 0.9069, sensitivity 0.9195, specificity 0.9564.

| strategy | clicks per image (pos + neg) | labelled pixels per image (%) | test dice | difference to full (Dice points) | % of full | test sens | test spec |
|---|---|---|---|---|---|---|---|
| center | 2 + 2 | 0.006 | 0.8346 | -7.2 | 92.0 | 0.7690 | 0.9703 |
| center | 5 + 5 | 0.015 | 0.8577 | -4.9 | 94.6 | 0.8454 | 0.9494 |
| center | 10 + 10 | 0.031 | 0.8626 | -4.4 | 95.1 | 0.8562 | 0.9481 |
| center | 5 + 20 | 0.038 | 0.8664 | -4.0 | 95.5 | 0.8275 | 0.9668 |
| center | 20 + 5 | 0.038 | 0.8583 | -4.9 | 94.6 | 0.8984 | 0.9216 |
| center | 20 + 20 | 0.061 | 0.8731 | -3.4 | 96.3 | 0.8972 | 0.9365 |
| center | 50 + 50 | 0.153 | 0.8805 | -2.6 | 97.1 | 0.9062 | 0.9388 |
| center | 100 + 100 | 0.305 | 0.8910 | -1.6 | 98.2 | 0.9013 | 0.9510 |
| random | 2 + 2 | 0.006 | 0.8467 | -6.0 | 93.4 | 0.8464 | 0.9385 |
| random | 5 + 5 | 0.015 | 0.8646 | -4.2 | 95.3 | 0.8298 | 0.9639 |
| random | 10 + 10 | 0.031 | 0.8633 | -4.4 | 95.2 | 0.8147 | 0.9708 |
| random | 20 + 20 | 0.061 | 0.8951 | -1.2 | 98.7 | 0.8964 | 0.9572 |
| random | 50 + 50 | 0.153 | 0.8956 | -1.1 | 98.7 | 0.8773 | 0.9671 |
| random | 100 + 100 | 0.305 | 0.8961 | -1.1 | 98.8 | 0.9123 | 0.9502 |

- within 1 Dice point(s) of full supervision: not reached by any run
- within 2 Dice point(s) of full supervision: from 20 + 20 clicks (random, test Dice 0.8951)
- within 3 Dice point(s) of full supervision: from 20 + 20 clicks (random, test Dice 0.8951)
- within 5 Dice point(s) of full supervision: from 5 + 5 clicks (center, test Dice 0.8577)
