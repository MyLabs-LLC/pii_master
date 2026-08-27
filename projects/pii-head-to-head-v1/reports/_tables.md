## Model scoreboard

| Measure | fusion-1k (arm A) | steady-cascade (arm B) | fusion-12k (arm C) |
| --- | ---: | ---: | ---: |
| Read window (chars) | 1,000 | 12,000 | 12,000 |
| **macro F2** (headline ranker) | 0.4800 | 0.6497 | 0.4809 |
| macro F1 | 0.4066 | 0.6235 | 0.4037 |
| macro F0.5 | 0.3712 | 0.6147 | 0.3663 |
| macro F3 | 0.5308 | 0.6682 | 0.5344 |
| macro precision | 0.3573 | 0.6154 | 0.3512 |
| macro recall | 0.7946 | 0.6482 | 0.8071 |
| micro F1 | 0.3652 | 0.7313 | 0.3647 |
| micro F2 | 0.5443 | 0.8025 | 0.5459 |
| micro F0.5 | 0.2797 | 0.6730 | 0.2788 |
| micro precision | 0.2429 | 0.6397 | 0.2418 |
| micro recall | 0.9031 | 0.8598 | 0.9152 |
| priority macro F0.5 (contra ranker) | 0.2060 | 0.7467 | 0.2054 |
| priority macro precision | 0.1829 | 0.7478 | 0.1824 |
| priority macro recall | 0.9991 | 0.7622 | 0.9991 |
| worst priority-tag recall | 0.9966 | 0.6524 | 0.9974 |
| document precision | 0.6163 | 0.8893 | 0.6163 |
| document recall | 1.0000 | 0.7532 | 1.0000 |
| document specificity | 0.0006 | 0.8832 | 0.0007 |
| prediction rate | 0.9999 | 0.7836 | 0.9999 |
| dead tag×corpus pairs (F2 = 0) | 8 | 16 | 6 |
| median tag F2 | 0.5020 | 0.7479 | 0.5030 |
| one-core p95 (ms) | 1.157 | 3.916 | 4.232 |
| one-core docs/s | 1361 | 314 | 270 |

## Per corpus

**macro F2 (the headline ranker)**

| Corpus | n | fusion-1k | steady-cascade | fusion-12k |
| --- | ---: | ---: | ---: | ---: |
| betterdataai_ner_silver_eval_10.36k | 10,360 | 0.2485 | 0.3122 | 0.2485 |
| ai4privacy_pii_masking_eval_10.63k | 10,626 | 0.5238 | 0.7476 | 0.5242 |
| pii_holdout_20.00k | 20,000 | 0.5512 | 0.7411 | 0.5699 |
| pii2_eval_25.15k | 30,000 | 0.4808 | 0.7822 | 0.4659 |
| openpii_pii_eval_38.94k | 38,937 | 0.5957 | 0.6656 | 0.5962 |
| datax-dualjudge-evalset-1.32k | 4,000 | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE |
| nemotron_eval_5.36k | 5,617 | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE |
| govdocs2-dualjudge-eval20-3.53k | 6,589 | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE |
| **equal-corpus mean** | | **0.4800** | **0.6497** | **0.4809** |

**micro F1**

| Corpus | n | fusion-1k | steady-cascade | fusion-12k |
| --- | ---: | ---: | ---: | ---: |
| betterdataai_ner_silver_eval_10.36k | 10,360 | 0.2108 | 0.6572 | 0.2108 |
| ai4privacy_pii_masking_eval_10.63k | 10,626 | 0.3477 | 0.6640 | 0.3477 |
| pii_holdout_20.00k | 20,000 | 0.4609 | 0.8177 | 0.4606 |
| pii2_eval_25.15k | 30,000 | 0.1890 | 0.6698 | 0.1862 |
| openpii_pii_eval_38.94k | 38,937 | 0.6178 | 0.8478 | 0.6181 |
| datax-dualjudge-evalset-1.32k | 4,000 | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE |
| nemotron_eval_5.36k | 5,617 | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE |
| govdocs2-dualjudge-eval20-3.53k | 6,589 | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE |
| **equal-corpus mean** | | **0.3652** | **0.7313** | **0.3647** |

**priority macro F0.5 (contra-view ranker)**

| Corpus | n | fusion-1k | steady-cascade | fusion-12k |
| --- | ---: | ---: | ---: | ---: |
| betterdataai_ner_silver_eval_10.36k | 10,360 | 0.2291 | 0.3597 | 0.2291 |
| ai4privacy_pii_masking_eval_10.63k | 10,626 | 0.1843 | 0.7686 | 0.1843 |
| pii_holdout_20.00k | 20,000 | 0.1855 | 0.8501 | 0.1833 |
| pii2_eval_25.15k | 30,000 | 0.0453 | 0.9272 | 0.0446 |
| openpii_pii_eval_38.94k | 38,937 | 0.3857 | 0.8279 | 0.3857 |
| datax-dualjudge-evalset-1.32k | 4,000 | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE |
| nemotron_eval_5.36k | 5,617 | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE |
| govdocs2-dualjudge-eval20-3.53k | 6,589 | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE |
| **equal-corpus mean** | | **0.2060** | **0.7467** | **0.2054** |

**macro recall**

| Corpus | n | fusion-1k | steady-cascade | fusion-12k |
| --- | ---: | ---: | ---: | ---: |
| betterdataai_ner_silver_eval_10.36k | 10,360 | 0.7796 | 0.4019 | 0.7796 |
| ai4privacy_pii_masking_eval_10.63k | 10,626 | 0.8147 | 0.8003 | 0.8152 |
| pii_holdout_20.00k | 20,000 | 0.7797 | 0.8346 | 0.8192 |
| pii2_eval_25.15k | 30,000 | 0.8323 | 0.8724 | 0.8499 |
| openpii_pii_eval_38.94k | 38,937 | 0.7875 | 0.6826 | 0.7881 |
| datax-dualjudge-evalset-1.32k | 4,000 | 0.7643 | 0.5772 | 0.7241 |
| nemotron_eval_5.36k | 5,617 | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE |
| govdocs2-dualjudge-eval20-3.53k | 6,589 | 0.8037 | 0.3685 | 0.8738 |
| **equal-corpus mean** | | **0.7946** | **0.6482** | **0.8071** |

**macro precision**

| Corpus | n | fusion-1k | steady-cascade | fusion-12k |
| --- | ---: | ---: | ---: | ---: |
| betterdataai_ner_silver_eval_10.36k | 10,360 | 0.1762 | 0.2292 | 0.1762 |
| ai4privacy_pii_masking_eval_10.63k | 10,626 | 0.3822 | 0.7362 | 0.3823 |
| pii_holdout_20.00k | 20,000 | 0.4248 | 0.7170 | 0.4186 |
| pii2_eval_25.15k | 30,000 | 0.3648 | 0.7345 | 0.3402 |
| openpii_pii_eval_38.94k | 38,937 | 0.4387 | 0.6603 | 0.4387 |
| datax-dualjudge-evalset-1.32k | 4,000 | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE |
| nemotron_eval_5.36k | 5,617 | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE |
| govdocs2-dualjudge-eval20-3.53k | 6,589 | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE |
| **equal-corpus mean** | | **0.3573** | **0.6154** | **0.3512** |

**prediction rate**

| Corpus | n | fusion-1k | steady-cascade | fusion-12k |
| --- | ---: | ---: | ---: | ---: |
| betterdataai_ner_silver_eval_10.36k | 10,360 | 1.0000 | 0.8882 | 1.0000 |
| ai4privacy_pii_masking_eval_10.63k | 10,626 | 1.0000 | 0.9189 | 1.0000 |
| pii_holdout_20.00k | 20,000 | 1.0000 | 0.9566 | 1.0000 |
| pii2_eval_25.15k | 30,000 | 0.9999 | 0.7941 | 0.9999 |
| openpii_pii_eval_38.94k | 38,937 | 1.0000 | 0.9779 | 1.0000 |
| datax-dualjudge-evalset-1.32k | 4,000 | 1.0000 | 0.3915 | 1.0000 |
| nemotron_eval_5.36k | 5,617 | 1.0000 | 0.9461 | 1.0000 |
| govdocs2-dualjudge-eval20-3.53k | 6,589 | 0.9994 | 0.3954 | 0.9992 |
| **equal-corpus mean** | | **0.9999** | **0.7836** | **0.9999** |

**betterdataai_ner_silver_eval_10.36k — worst 20 tags by F2, per arm**

*fusion-1k (arm A)*

| tag | n | F2 | precision | recall | predicted | state |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sensitive_pii_social_security_number` ★ | 1 | 0.0006 | 0.0001 | 1.0000 | 9,047 |  |
| `sensitive_pii_passport_number` ★ | 1 | 0.0007 | 0.0001 | 1.0000 | 7,242 |  |
| `sensitive_phi_health_plan_beneficiary_number` ★ | 1 | 0.0007 | 0.0001 | 1.0000 | 7,196 |  |
| `sensitive_pci_individual_taxpayer_identification_number_itin` ★ | 1 | 0.0009 | 0.0002 | 1.0000 | 5,679 |  |
| `sensitive_pci_credit_card_number` ★ | 2 | 0.0015 | 0.0003 | 1.0000 | 6,641 |  |
| `sensitive_pci_bank_account_number` ★ | 5 | 0.0031 | 0.0006 | 0.6000 | 4,859 |  |
| `sensitive_phi_patient_id_number` ★ | 28 | 0.0225 | 0.0046 | 0.8929 | 5,447 |  |
| `sensitive_pii_password` ★ | 50 | 0.0237 | 0.0048 | 1.0000 | 10,354 |  |
| `sensitive_pii_email` | 134 | 0.0333 | 0.0625 | 0.0299 | 64 |  |
| `sensitive_phi_medical_record_number_mrn` ★ | 151 | 0.0793 | 0.0170 | 0.9868 | 8,788 |  |
| `sensitive_pii_address` ★ | 254 | 0.1120 | 0.0246 | 1.0000 | 10,326 |  |
| `sensitive_pii_zip_code` | 36 | 0.2482 | 0.0753 | 0.5833 | 279 |  |
| `sensitive_pii_phone_number` | 47 | 0.2518 | 0.1556 | 0.2979 | 90 |  |
| `sensitive_pii_date_of_birth_dob` | 460 | 0.3136 | 0.1714 | 0.3957 | 1,062 |  |
| `sensitive_pci_credit_card` | 2 | 0.3846 | 0.2000 | 0.5000 | 5 |  |
| `sensitive_pii_county` | 387 | 0.6074 | 0.4263 | 0.6796 | 617 |  |
| `sensitive_pii_state` | 1,565 | 0.6764 | 0.3744 | 0.8473 | 3,542 |  |
| `sensitive_pii_full_name` ★ | 8,592 | 0.9605 | 0.8293 | 1.0000 | 10,360 |  |
| `sensitive_pii_vehicle_identification_number_vin` | 1 | 1.0000 | 1.0000 | 1.0000 | 1 |  |

*steady-cascade (arm B)*

| tag | n | F2 | precision | recall | predicted | state |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sensitive_phi_patient_id_number` ★ | 28 | 0.0000 | NOT MEASURABLE | 0.0000 | 0 | **unlearned** |
| `sensitive_pci_bank_account_number` ★ | 5 | 0.0000 | 0.0000 | 0.0000 | 1 |  |
| `sensitive_pci_credit_card` | 2 | 0.0000 | 0.0000 | 0.0000 | 2 |  |
| `sensitive_pci_credit_card_number` ★ | 2 | 0.0000 | 0.0000 | 0.0000 | 2 |  |
| `sensitive_pci_individual_taxpayer_identification_number_itin` ★ | 1 | 0.0000 | NOT MEASURABLE | 0.0000 | 0 | **unlearned** |
| `sensitive_phi_health_plan_beneficiary_number` ★ | 1 | 0.0000 | NOT MEASURABLE | 0.0000 | 0 | **unlearned** |
| `sensitive_pii_social_security_number` ★ | 1 | 0.0000 | NOT MEASURABLE | 0.0000 | 0 | **unlearned** |
| `sensitive_pii_vehicle_identification_number_vin` | 1 | 0.0000 | 0.0000 | 0.0000 | 9 |  |
| `sensitive_pii_password` ★ | 50 | 0.0731 | 0.0174 | 0.3600 | 1,032 |  |
| `sensitive_pii_county` | 387 | 0.3302 | 0.0943 | 0.8811 | 3,615 |  |
| `sensitive_phi_medical_record_number_mrn` ★ | 151 | 0.3949 | 0.2968 | 0.4305 | 219 |  |
| `sensitive_pii_address` ★ | 254 | 0.3968 | 0.1526 | 0.6614 | 1,101 |  |
| `sensitive_pii_zip_code` | 36 | 0.4134 | 0.1909 | 0.5833 | 110 |  |
| `sensitive_pii_date_of_birth_dob` | 460 | 0.5413 | 0.2637 | 0.7348 | 1,282 |  |
| `sensitive_pii_email` | 134 | 0.6431 | 0.5705 | 0.6642 | 156 |  |
| `sensitive_pii_state` | 1,565 | 0.6806 | 0.6541 | 0.6875 | 1,645 |  |
| `sensitive_pii_phone_number` | 47 | 0.7613 | 0.6727 | 0.7872 | 55 |  |
| `sensitive_pii_passport_number` ★ | 1 | 0.8333 | 0.5000 | 1.0000 | 2 |  |
| `sensitive_pii_full_name` ★ | 8,592 | 0.8633 | 0.9413 | 0.8458 | 7,720 |  |

*fusion-12k (arm C)*

| tag | n | F2 | precision | recall | predicted | state |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sensitive_pii_social_security_number` ★ | 1 | 0.0006 | 0.0001 | 1.0000 | 9,047 |  |
| `sensitive_pii_passport_number` ★ | 1 | 0.0007 | 0.0001 | 1.0000 | 7,242 |  |
| `sensitive_phi_health_plan_beneficiary_number` ★ | 1 | 0.0007 | 0.0001 | 1.0000 | 7,196 |  |
| `sensitive_pci_individual_taxpayer_identification_number_itin` ★ | 1 | 0.0009 | 0.0002 | 1.0000 | 5,679 |  |
| `sensitive_pci_credit_card_number` ★ | 2 | 0.0015 | 0.0003 | 1.0000 | 6,641 |  |
| `sensitive_pci_bank_account_number` ★ | 5 | 0.0031 | 0.0006 | 0.6000 | 4,859 |  |
| `sensitive_phi_patient_id_number` ★ | 28 | 0.0225 | 0.0046 | 0.8929 | 5,447 |  |
| `sensitive_pii_password` ★ | 50 | 0.0237 | 0.0048 | 1.0000 | 10,354 |  |
| `sensitive_pii_email` | 134 | 0.0333 | 0.0625 | 0.0299 | 64 |  |
| `sensitive_phi_medical_record_number_mrn` ★ | 151 | 0.0793 | 0.0170 | 0.9868 | 8,788 |  |
| `sensitive_pii_address` ★ | 254 | 0.1120 | 0.0246 | 1.0000 | 10,326 |  |
| `sensitive_pii_zip_code` | 36 | 0.2482 | 0.0753 | 0.5833 | 279 |  |
| `sensitive_pii_phone_number` | 47 | 0.2518 | 0.1556 | 0.2979 | 90 |  |
| `sensitive_pii_date_of_birth_dob` | 460 | 0.3136 | 0.1714 | 0.3957 | 1,062 |  |
| `sensitive_pci_credit_card` | 2 | 0.3846 | 0.2000 | 0.5000 | 5 |  |
| `sensitive_pii_county` | 387 | 0.6074 | 0.4263 | 0.6796 | 617 |  |
| `sensitive_pii_state` | 1,565 | 0.6764 | 0.3744 | 0.8473 | 3,542 |  |
| `sensitive_pii_full_name` ★ | 8,592 | 0.9605 | 0.8293 | 1.0000 | 10,360 |  |
| `sensitive_pii_vehicle_identification_number_vin` | 1 | 1.0000 | 1.0000 | 1.0000 | 1 |  |

★ = priority tag. **unlearned** = never predicted once; *silent* = recognised but almost never emitted.

**ai4privacy_pii_masking_eval_10.63k — worst 20 tags by F2, per arm**

*fusion-1k (arm A)*

| tag | n | F2 | precision | recall | predicted | state |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sensitive_pii_personal_identification_number_pin` ★ | 145 | 0.0724 | 0.0154 | 1.0000 | 9,438 |  |
| `sensitive_pci_credit_card_number` ★ | 524 | 0.2268 | 0.0554 | 1.0000 | 9,456 |  |
| `sensitive_pci_iban` ★ | 431 | 0.3011 | 0.0793 | 1.0000 | 5,432 |  |
| `sensitive_pci_bank_account_number` ★ | 531 | 0.3300 | 0.0897 | 1.0000 | 5,921 |  |
| `sensitive_pii_vehicle_identification_number_vin` | 80 | 0.3511 | 0.6944 | 0.3125 | 36 |  |
| `sensitive_pii_driver_s_license_number` ★ | 1,101 | 0.3753 | 0.1073 | 1.0000 | 10,265 |  |
| `sensitive_pii_passport_number` ★ | 1,102 | 0.3899 | 0.1134 | 0.9991 | 9,712 |  |
| `sensitive_pii_password` ★ | 1,245 | 0.3992 | 0.1173 | 1.0000 | 10,614 |  |
| `sensitive_pii_ipv6` | 250 | 0.4010 | 0.1896 | 0.5560 | 733 |  |
| `sensitive_pci_swift_code` | 130 | 0.4062 | 0.1579 | 0.6692 | 551 |  |
| `sensitive_pii_date_of_birth_dob` | 1,233 | 0.5083 | 0.4380 | 0.5296 | 1,491 |  |
| `sensitive_pii_phone_number` | 1,228 | 0.5198 | 0.6548 | 0.4943 | 927 |  |
| `sensitive_pci_card_verification_value_cvv` | 154 | 0.5491 | 1.0000 | 0.4935 | 76 |  |
| `sensitive_pii_state` | 1,721 | 0.5494 | 0.4524 | 0.5805 | 2,208 |  |
| `sensitive_pci_last_four_credit_card_number_digits` | 245 | 0.5935 | 0.2446 | 0.9224 | 924 |  |
| `sensitive_pii_address` ★ | 2,529 | 0.6106 | 0.2387 | 1.0000 | 10,593 |  |
| `sensitive_pii_social_security_number` ★ | 2,400 | 0.6153 | 0.2425 | 0.9992 | 9,888 |  |
| `sensitive_pii_county` | 536 | 0.6497 | 0.8479 | 0.6138 | 388 |  |
| `sensitive_pii_mac_address` | 108 | 0.6997 | 0.4363 | 0.8241 | 204 |  |
| `sensitive_pii_zip_code` | 1,635 | 0.7286 | 0.7127 | 0.7327 | 1,681 |  |

*steady-cascade (arm B)*

| tag | n | F2 | precision | recall | predicted | state |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sensitive_pii_county` | 536 | 0.3229 | 0.0903 | 0.9067 | 5,382 |  |
| `sensitive_pii_password` ★ | 1,245 | 0.4907 | 0.1699 | 0.9293 | 6,810 |  |
| `sensitive_pii_vehicle_identification_number_vin` | 80 | 0.4965 | 0.3805 | 0.5375 | 113 |  |
| `sensitive_pci_swift_code` | 130 | 0.6071 | 0.3187 | 0.7846 | 320 |  |
| `sensitive_pci_last_four_credit_card_number_digits` | 245 | 0.6623 | 0.4619 | 0.7429 | 394 |  |
| `sensitive_pii_full_name` ★ | 5,470 | 0.7044 | 0.7837 | 0.6870 | 4,795 |  |
| `sensitive_pii_personal_identification_number_pin` ★ | 145 | 0.7216 | 0.9340 | 0.6828 | 106 |  |
| `sensitive_pii_ipv6` | 250 | 0.7392 | 0.6113 | 0.7800 | 319 |  |
| `sensitive_pii_state` | 1,721 | 0.7431 | 0.8596 | 0.7188 | 1,439 |  |
| `sensitive_pii_phone_number` | 1,228 | 0.7577 | 0.8908 | 0.7305 | 1,007 |  |
| `sensitive_pii_social_security_number` ★ | 2,400 | 0.7601 | 0.9482 | 0.7242 | 1,833 |  |
| `sensitive_pci_card_verification_value_cvv` | 154 | 0.7661 | 0.9739 | 0.7273 | 115 |  |
| `sensitive_pii_date_of_birth_dob` | 1,233 | 0.7819 | 0.6702 | 0.8159 | 1,501 |  |
| `sensitive_pii_passport_number` ★ | 1,102 | 0.7918 | 0.8089 | 0.7877 | 1,073 |  |
| `sensitive_pii_driver_s_license_number` ★ | 1,101 | 0.8008 | 0.9203 | 0.7757 | 928 |  |
| `sensitive_pii_address` ★ | 2,529 | 0.8116 | 0.5284 | 0.9371 | 4,485 |  |
| `sensitive_pci_bank_account_number` ★ | 531 | 0.8127 | 0.9787 | 0.7797 | 423 |  |
| `sensitive_pci_iban` ★ | 431 | 0.8147 | 0.9941 | 0.7796 | 338 |  |
| `sensitive_pii_ipv4` | 1,346 | 0.8489 | 0.9578 | 0.8254 | 1,160 |  |
| `sensitive_pci_credit_card` | 524 | 0.8615 | 0.8149 | 0.8740 | 562 |  |

*fusion-12k (arm C)*

| tag | n | F2 | precision | recall | predicted | state |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sensitive_pii_personal_identification_number_pin` ★ | 145 | 0.0724 | 0.0154 | 1.0000 | 9,438 |  |
| `sensitive_pci_credit_card_number` ★ | 524 | 0.2268 | 0.0554 | 1.0000 | 9,456 |  |
| `sensitive_pci_iban` ★ | 431 | 0.3011 | 0.0793 | 1.0000 | 5,433 |  |
| `sensitive_pci_bank_account_number` ★ | 531 | 0.3300 | 0.0897 | 1.0000 | 5,921 |  |
| `sensitive_pii_vehicle_identification_number_vin` | 80 | 0.3511 | 0.6944 | 0.3125 | 36 |  |
| `sensitive_pii_driver_s_license_number` ★ | 1,101 | 0.3753 | 0.1073 | 1.0000 | 10,265 |  |
| `sensitive_pii_passport_number` ★ | 1,102 | 0.3899 | 0.1134 | 0.9991 | 9,712 |  |
| `sensitive_pii_password` ★ | 1,245 | 0.3992 | 0.1173 | 1.0000 | 10,614 |  |
| `sensitive_pii_ipv6` | 250 | 0.4010 | 0.1896 | 0.5560 | 733 |  |
| `sensitive_pci_swift_code` | 130 | 0.4062 | 0.1579 | 0.6692 | 551 |  |
| `sensitive_pii_date_of_birth_dob` | 1,233 | 0.5077 | 0.4379 | 0.5288 | 1,489 |  |
| `sensitive_pii_phone_number` | 1,228 | 0.5221 | 0.6559 | 0.4967 | 930 |  |
| `sensitive_pii_state` | 1,721 | 0.5494 | 0.4524 | 0.5805 | 2,208 |  |
| `sensitive_pci_card_verification_value_cvv` | 154 | 0.5556 | 1.0000 | 0.5000 | 77 |  |
| `sensitive_pci_last_four_credit_card_number_digits` | 245 | 0.5932 | 0.2443 | 0.9224 | 925 |  |
| `sensitive_pii_address` ★ | 2,529 | 0.6106 | 0.2387 | 1.0000 | 10,593 |  |
| `sensitive_pii_social_security_number` ★ | 2,400 | 0.6153 | 0.2425 | 0.9992 | 9,888 |  |
| `sensitive_pii_county` | 536 | 0.6480 | 0.8475 | 0.6119 | 387 |  |
| `sensitive_pii_mac_address` | 108 | 0.6997 | 0.4363 | 0.8241 | 204 |  |
| `sensitive_pii_zip_code` | 1,635 | 0.7291 | 0.7124 | 0.7333 | 1,683 |  |

★ = priority tag. **unlearned** = never predicted once; *silent* = recognised but almost never emitted.

**pii_holdout_20.00k — worst 20 tags by F2, per arm**

*fusion-1k (arm A)*

| tag | n | F2 | precision | recall | predicted | state |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sensitive_phi_medical_treatment` | 22 | 0.0000 | NOT MEASURABLE | 0.0000 | 0 | **unlearned** |
| `sensitive_pii_income` | 10 | 0.0000 | 0.0000 | 0.0000 | 207 |  |
| `sensitive_phi_medication` | 9 | 0.0000 | NOT MEASURABLE | 0.0000 | 0 | **unlearned** |
| `sensitive_phi_patient_id_number` ★ | 1 | 0.0003 | 0.0001 | 1.0000 | 18,297 |  |
| `sensitive_phi_medical_condition` | 60 | 0.0176 | 0.0227 | 0.0167 | 44 |  |
| `sensitive_pii_country_of_origin` | 9 | 0.0204 | 0.0044 | 0.2222 | 455 |  |
| `sensitive_pii_password` ★ | 232 | 0.0554 | 0.0116 | 1.0000 | 19,998 |  |
| `sensitive_pii_personal_identification_number_pin` ★ | 286 | 0.0688 | 0.0146 | 1.0000 | 19,626 |  |
| `sensitive_phi_health_plan_beneficiary_number` ★ | 354 | 0.0864 | 0.0186 | 1.0000 | 19,065 |  |
| `sensitive_phi_medical_record_number_mrn` ★ | 418 | 0.0999 | 0.0217 | 1.0000 | 19,239 |  |
| `sensitive_pii_marital_status` | 11 | 0.1020 | 0.2000 | 0.0909 | 5 |  |
| `sensitive_pci_bank_account_number` ★ | 591 | 0.1576 | 0.0361 | 0.9983 | 16,351 |  |
| `sensitive_pii_geolocation` | 225 | 0.3750 | 0.1105 | 0.9333 | 1,900 |  |
| `sensitive_pci_credit_card_number` ★ | 2,415 | 0.4142 | 0.1239 | 1.0000 | 19,493 |  |
| `sensitive_pii_username` | 313 | 0.4335 | 0.1795 | 0.6709 | 1,170 |  |
| `sensitive_pci_individual_taxpayer_identification_number_itin` ★ | 2,637 | 0.4384 | 0.1350 | 1.0000 | 19,528 |  |
| `sensitive_pii_employment_status` | 574 | 0.4461 | 0.2225 | 0.5958 | 1,537 |  |
| `sensitive_pii_driver_s_license_number` ★ | 2,808 | 0.4527 | 0.1420 | 1.0000 | 19,779 |  |
| `sensitive_pii_sexual_identity_and_orientation` | 142 | 0.4669 | 0.2312 | 0.6268 | 385 |  |
| `sensitive_pii_social_security_number` ★ | 3,166 | 0.4897 | 0.1610 | 1.0000 | 19,661 |  |

*steady-cascade (arm B)*

| tag | n | F2 | precision | recall | predicted | state |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sensitive_phi_patient_id_number` ★ | 1 | 0.0000 | NOT MEASURABLE | 0.0000 | 0 | **unlearned** |
| `sensitive_pii_sexual_identity_and_orientation` | 142 | 0.1031 | 0.0225 | 0.9718 | 6,126 |  |
| `sensitive_pii_country_of_origin` | 9 | 0.1154 | 0.0319 | 0.3333 | 94 |  |
| `sensitive_pii_geolocation` | 225 | 0.2340 | 0.0585 | 0.9333 | 3,587 |  |
| `sensitive_pii_religion` | 272 | 0.2354 | 0.0587 | 0.9485 | 4,392 |  |
| `sensitive_phi_medication` | 9 | 0.2632 | 1.0000 | 0.2222 | 2 | silent |
| `sensitive_pii_marital_status` | 11 | 0.3191 | 1.0000 | 0.2727 | 3 | silent |
| `sensitive_phi_medical_treatment` | 22 | 0.3646 | 0.8750 | 0.3182 | 8 |  |
| `sensitive_pii_ethnicity` | 513 | 0.4327 | 0.1462 | 0.8480 | 2,975 |  |
| `sensitive_phi_medical_condition` | 60 | 0.4732 | 0.3896 | 0.5000 | 77 |  |
| `sensitive_pii_username` | 313 | 0.4909 | 0.2079 | 0.7444 | 1,121 |  |
| `sensitive_pii_county` | 407 | 0.5104 | 0.1786 | 0.9533 | 2,173 |  |
| `sensitive_pii_income` | 10 | 0.5882 | 0.5455 | 0.6000 | 11 |  |
| `sensitive_pii_password` ★ | 232 | 0.6661 | 0.2957 | 0.9698 | 761 |  |
| `sensitive_pii_employment_status` | 574 | 0.6681 | 0.3574 | 0.8537 | 1,371 |  |
| `sensitive_pii_country_of_residence` | 1,270 | 0.7236 | 0.3771 | 0.9394 | 3,164 |  |
| `sensitive_pii_educational_level` | 427 | 0.7353 | 0.4058 | 0.9227 | 971 |  |
| `sensitive_pii_military_identification_number` ★ | 4,327 | 0.7548 | 0.3814 | 0.9995 | 11,341 |  |
| `sensitive_pii_driver_s_license_number` ★ | 2,808 | 0.8153 | 0.8518 | 0.8066 | 2,659 |  |
| `sensitive_pci_financial_instrument_global_identifier_figi` | 2,808 | 0.8229 | 0.8493 | 0.8166 | 2,700 |  |

*fusion-12k (arm C)*

| tag | n | F2 | precision | recall | predicted | state |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sensitive_phi_medical_treatment` | 22 | 0.0000 | NOT MEASURABLE | 0.0000 | 0 | **unlearned** |
| `sensitive_phi_medication` | 9 | 0.0000 | NOT MEASURABLE | 0.0000 | 0 | **unlearned** |
| `sensitive_phi_patient_id_number` ★ | 1 | 0.0003 | 0.0001 | 1.0000 | 19,308 |  |
| `sensitive_pii_country_of_origin` | 9 | 0.0272 | 0.0058 | 0.3333 | 515 |  |
| `sensitive_phi_medical_condition` | 60 | 0.0328 | 0.0308 | 0.0333 | 65 |  |
| `sensitive_pii_income` | 10 | 0.0482 | 0.0111 | 0.3000 | 271 |  |
| `sensitive_pii_password` ★ | 232 | 0.0554 | 0.0116 | 1.0000 | 19,999 |  |
| `sensitive_pii_personal_identification_number_pin` ★ | 286 | 0.0677 | 0.0143 | 1.0000 | 19,967 |  |
| `sensitive_phi_health_plan_beneficiary_number` ★ | 354 | 0.0838 | 0.0180 | 1.0000 | 19,718 |  |
| `sensitive_phi_medical_record_number_mrn` ★ | 418 | 0.0972 | 0.0211 | 1.0000 | 19,832 |  |
| `sensitive_pii_marital_status` | 11 | 0.0980 | 0.1429 | 0.0909 | 7 |  |
| `sensitive_pci_bank_account_number` ★ | 591 | 0.1401 | 0.0315 | 1.0000 | 18,734 |  |
| `sensitive_pii_geolocation` | 225 | 0.3170 | 0.0863 | 0.9556 | 2,491 |  |
| `sensitive_pci_credit_card_number` ★ | 2,415 | 0.4091 | 0.1216 | 1.0000 | 19,857 |  |
| `sensitive_pii_username` | 313 | 0.4255 | 0.1552 | 0.7540 | 1,521 |  |
| `sensitive_pci_individual_taxpayer_identification_number_itin` ★ | 2,637 | 0.4332 | 0.1326 | 1.0000 | 19,891 |  |
| `sensitive_pii_driver_s_license_number` ★ | 2,808 | 0.4496 | 0.1404 | 1.0000 | 19,996 |  |
| `sensitive_pii_social_security_number` ★ | 3,166 | 0.4848 | 0.1584 | 1.0000 | 19,990 |  |
| `sensitive_pii_employment_status` | 574 | 0.4859 | 0.2108 | 0.7213 | 1,964 |  |
| `sensitive_pii_sexual_identity_and_orientation` | 142 | 0.4916 | 0.2227 | 0.7042 | 449 |  |

★ = priority tag. **unlearned** = never predicted once; *silent* = recognised but almost never emitted.

**pii2_eval_25.15k — worst 20 tags by F2, per arm**

*fusion-1k (arm A)*

| tag | n | F2 | precision | recall | predicted | state |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sensitive_pii_date_of_death` | 15 | 0.0000 | 0.0000 | 0.0000 | 407 |  |
| `sensitive_pii_military_identification_number` ★ | 1 | 0.0002 | 0.0000 | 1.0000 | 29,141 |  |
| `sensitive_pii_personal_identification_number_pin` ★ | 182 | 0.0316 | 0.0065 | 1.0000 | 28,051 |  |
| `sensitive_pii_password` ★ | 196 | 0.0325 | 0.0067 | 1.0000 | 29,413 |  |
| `sensitive_pii_driver_s_license_number` ★ | 219 | 0.0360 | 0.0074 | 1.0000 | 29,548 |  |
| `sensitive_pii_passport_number` ★ | 227 | 0.0377 | 0.0078 | 1.0000 | 29,182 |  |
| `sensitive_pii_visa_number` ★ | 227 | 0.0378 | 0.0078 | 1.0000 | 29,140 |  |
| `sensitive_pii_social_security_number` ★ | 237 | 0.0390 | 0.0080 | 0.9958 | 29,340 |  |
| `sensitive_pci_individual_taxpayer_identification_number_itin` ★ | 234 | 0.0391 | 0.0081 | 1.0000 | 28,997 |  |
| `sensitive_pii_ethnicity` | 340 | 0.0413 | 0.1304 | 0.0353 | 92 |  |
| `sensitive_pii_geolocation` | 37 | 0.0436 | 0.0097 | 0.3514 | 1,344 |  |
| `sensitive_pci_credit_card_number` ★ | 309 | 0.0517 | 0.0108 | 1.0000 | 28,620 |  |
| `sensitive_phi_patient_id_number` ★ | 326 | 0.0605 | 0.0127 | 1.0000 | 25,651 |  |
| `sensitive_pii_sexual_identity_and_orientation` | 54 | 0.0625 | 0.0385 | 0.0741 | 104 |  |
| `sensitive_pci_iban` ★ | 309 | 0.0683 | 0.0145 | 1.0000 | 21,384 |  |
| `sensitive_pci_bank_account_number` ★ | 316 | 0.0740 | 0.0157 | 1.0000 | 20,098 |  |
| `sensitive_phi_medical_record_number_mrn` ★ | 535 | 0.0884 | 0.0190 | 1.0000 | 28,124 |  |
| `sensitive_phi_health_plan_beneficiary_number` ★ | 536 | 0.0901 | 0.0194 | 1.0000 | 27,600 |  |
| `sensitive_pii_gender_and_sex` | 28 | 0.1378 | 0.0493 | 0.2500 | 142 |  |
| `sensitive_pii_religion` | 46 | 0.1546 | 0.0841 | 0.1957 | 107 |  |

*steady-cascade (arm B)*

| tag | n | F2 | precision | recall | predicted | state |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sensitive_pii_military_identification_number` ★ | 1 | 0.0004 | 0.0001 | 1.0000 | 13,440 |  |
| `sensitive_pii_date_of_death` | 15 | 0.0045 | 0.0009 | 0.6000 | 9,952 |  |
| `sensitive_pii_geolocation` | 37 | 0.0246 | 0.0052 | 0.3243 | 2,287 |  |
| `sensitive_pii_sexual_identity_and_orientation` | 54 | 0.0299 | 0.0062 | 0.6481 | 5,644 |  |
| `sensitive_pii_gender_and_sex` | 28 | 0.0377 | 0.0080 | 0.5000 | 1,747 |  |
| `sensitive_pii_religion` | 46 | 0.1001 | 0.0226 | 0.6957 | 1,414 |  |
| `sensitive_pii_ethnicity` | 340 | 0.1713 | 0.1742 | 0.1706 | 333 |  |
| `sensitive_pii_county` | 161 | 0.3528 | 0.1394 | 0.5714 | 660 |  |
| `sensitive_pii_employment_status` | 452 | 0.5613 | 0.2990 | 0.7190 | 1,087 |  |
| `sensitive_pii_username` | 298 | 0.6225 | 0.3791 | 0.7416 | 583 |  |
| `sensitive_pii_country_of_residence` | 495 | 0.6784 | 0.4953 | 0.7475 | 747 |  |
| `sensitive_pii_country_of_origin` | 240 | 0.7247 | 0.3719 | 0.9500 | 613 |  |
| `sensitive_pii_state` | 3,903 | 0.7507 | 0.7206 | 0.7586 | 4,109 |  |
| `sensitive_pii_area_code` | 4,921 | 0.7643 | 0.8762 | 0.7407 | 4,160 |  |
| `sensitive_pii_password` ★ | 196 | 0.7766 | 0.4452 | 0.9541 | 420 |  |
| `sensitive_pii_educational_level` | 312 | 0.7821 | 0.5392 | 0.8814 | 510 |  |
| `sensitive_pii_address` ★ | 2,841 | 0.7879 | 0.6918 | 0.8163 | 3,352 |  |
| `sensitive_phi_medical_condition` | 815 | 0.7890 | 0.7231 | 0.8074 | 910 |  |
| `sensitive_pii_phone_number` | 6,908 | 0.8213 | 0.8919 | 0.8053 | 6,237 |  |
| `sensitive_pii_zip_code` | 5,031 | 0.8304 | 0.7701 | 0.8469 | 5,533 |  |

*fusion-12k (arm C)*

| tag | n | F2 | precision | recall | predicted | state |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sensitive_pii_military_identification_number` ★ | 1 | 0.0002 | 0.0000 | 1.0000 | 29,653 |  |
| `sensitive_pii_date_of_death` | 15 | 0.0064 | 0.0014 | 0.0667 | 723 |  |
| `sensitive_pii_ethnicity` | 340 | 0.0069 | 0.0250 | 0.0059 | 80 |  |
| `sensitive_pii_personal_identification_number_pin` ★ | 182 | 0.0310 | 0.0064 | 1.0000 | 28,623 |  |
| `sensitive_pii_password` ★ | 196 | 0.0325 | 0.0067 | 1.0000 | 29,407 |  |
| `sensitive_pii_driver_s_license_number` ★ | 219 | 0.0356 | 0.0073 | 1.0000 | 29,923 |  |
| `sensitive_pii_passport_number` ★ | 227 | 0.0371 | 0.0076 | 1.0000 | 29,710 |  |
| `sensitive_pii_visa_number` ★ | 227 | 0.0371 | 0.0077 | 1.0000 | 29,645 |  |
| `sensitive_pii_social_security_number` ★ | 237 | 0.0386 | 0.0080 | 1.0000 | 29,777 |  |
| `sensitive_pii_geolocation` | 37 | 0.0394 | 0.0083 | 0.5676 | 2,519 |  |
| `sensitive_pci_individual_taxpayer_identification_number_itin` ★ | 234 | 0.0395 | 0.0082 | 1.0000 | 28,706 |  |
| `sensitive_pci_credit_card_number` ★ | 309 | 0.0522 | 0.0109 | 1.0000 | 28,367 |  |
| `sensitive_phi_patient_id_number` ★ | 326 | 0.0562 | 0.0118 | 1.0000 | 27,680 |  |
| `sensitive_pci_iban` ★ | 309 | 0.0573 | 0.0120 | 1.0000 | 25,722 |  |
| `sensitive_pci_bank_account_number` ★ | 316 | 0.0597 | 0.0125 | 1.0000 | 25,189 |  |
| `sensitive_pii_sexual_identity_and_orientation` | 54 | 0.0732 | 0.0309 | 0.1111 | 194 |  |
| `sensitive_phi_medical_record_number_mrn` ★ | 535 | 0.0852 | 0.0183 | 1.0000 | 29,247 |  |
| `sensitive_phi_health_plan_beneficiary_number` ★ | 536 | 0.0865 | 0.0186 | 1.0000 | 28,854 |  |
| `sensitive_pii_gender_and_sex` | 28 | 0.0988 | 0.0273 | 0.2857 | 293 |  |
| `sensitive_pii_religion` | 46 | 0.1174 | 0.0614 | 0.1522 | 114 |  |

★ = priority tag. **unlearned** = never predicted once; *silent* = recognised but almost never emitted.

**openpii_pii_eval_38.94k — worst 20 tags by F2, per arm**

*fusion-1k (arm A)*

| tag | n | F2 | precision | recall | predicted | state |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sensitive_pii_country_of_origin` | 3 | 0.0000 | 0.0000 | 0.0000 | 15 |  |
| `sensitive_pii_country_of_residence` | 3 | 0.0000 | 0.0000 | 0.0000 | 10 |  |
| `sensitive_pii_nationality` | 3 | 0.0000 | 0.0000 | 0.0000 | 7 |  |
| `sensitive_pii_username` | 2 | 0.0000 | 0.0000 | 0.0000 | 35 |  |
| `sensitive_pci_iban` ★ | 1 | 0.0002 | 0.0000 | 1.0000 | 24,262 |  |
| `sensitive_pci_bank_account_number` ★ | 1 | 0.0002 | 0.0000 | 1.0000 | 23,665 |  |
| `sensitive_pii_ipv4` | 2 | 0.2000 | 0.0588 | 0.5000 | 17 |  |
| `sensitive_pii_social_security_number` ★ | 6,748 | 0.5146 | 0.1749 | 1.0000 | 38,579 |  |
| `sensitive_pci_individual_taxpayer_identification_number_itin` ★ | 7,672 | 0.5568 | 0.2008 | 1.0000 | 38,209 |  |
| `sensitive_pii_driver_s_license_number` ★ | 8,538 | 0.5845 | 0.2196 | 0.9999 | 38,870 |  |
| `sensitive_pci_financial_instrument_global_identifier_figi` | 8,538 | 0.5941 | 0.3025 | 0.7827 | 22,096 |  |
| `sensitive_pci_credit_card_number` ★ | 8,908 | 0.6005 | 0.2311 | 1.0000 | 38,542 |  |
| `sensitive_pii_passport_number` ★ | 13,763 | 0.7337 | 0.3554 | 0.9997 | 38,710 |  |
| `sensitive_pii_visa_number` ★ | 13,763 | 0.7339 | 0.3558 | 0.9995 | 38,664 |  |
| `sensitive_pii_military_identification_number` ★ | 13,763 | 0.7339 | 0.3558 | 0.9996 | 38,667 |  |
| `sensitive_pii_last_four_us_ssn_digits` | 6,748 | 0.7470 | 0.6712 | 0.7687 | 7,728 |  |
| `sensitive_pii_immigration_and_citizenship_status` | 13,763 | 0.7545 | 0.4763 | 0.8835 | 25,528 |  |
| `sensitive_pii_address` ★ | 15,267 | 0.7633 | 0.3921 | 1.0000 | 38,933 |  |
| `sensitive_pii_age` | 13,255 | 0.7956 | 0.5485 | 0.8966 | 21,666 |  |
| `sensitive_pii_area_code` | 15,314 | 0.7971 | 0.7559 | 0.8081 | 16,371 |  |

*steady-cascade (arm B)*

| tag | n | F2 | precision | recall | predicted | state |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sensitive_pii_country_of_origin` | 3 | 0.0000 | 0.0000 | 0.0000 | 3 |  |
| `sensitive_pii_country_of_residence` | 3 | 0.0000 | 0.0000 | 0.0000 | 63 |  |
| `sensitive_pii_nationality` | 3 | 0.0000 | NOT MEASURABLE | 0.0000 | 0 | **unlearned** |
| `sensitive_pii_ipv4` | 2 | 0.0000 | 0.0000 | 0.0000 | 1 |  |
| `sensitive_pii_username` | 2 | 0.0000 | 0.0000 | 0.0000 | 28 |  |
| `sensitive_pci_bank_account_number` ★ | 1 | 0.0000 | 0.0000 | 0.0000 | 3 |  |
| `sensitive_pci_iban` ★ | 1 | 0.0000 | NOT MEASURABLE | 0.0000 | 0 | **unlearned** |
| `sensitive_pii_gender_and_sex` | 10,002 | 0.6693 | 0.2904 | 0.9933 | 34,210 |  |
| `sensitive_pii_driver_s_license_number` ★ | 8,538 | 0.7116 | 0.7655 | 0.6993 | 7,800 |  |
| `sensitive_pci_financial_instrument_global_identifier_figi` | 8,538 | 0.7144 | 0.7574 | 0.7044 | 7,940 |  |
| `sensitive_pii_military_identification_number` ★ | 13,763 | 0.7310 | 0.3576 | 0.9892 | 38,077 |  |
| `sensitive_pii_visa_number` ★ | 13,763 | 0.7961 | 0.9194 | 0.7703 | 11,532 |  |
| `sensitive_pii_immigration_and_citizenship_status` | 13,763 | 0.7975 | 0.9176 | 0.7722 | 11,583 |  |
| `sensitive_pii_last_four_us_ssn_digits` | 6,748 | 0.8149 | 0.9388 | 0.7888 | 5,670 |  |
| `sensitive_pii_passport_number` ★ | 13,763 | 0.8360 | 0.8949 | 0.8224 | 12,649 |  |
| `sensitive_pii_age` | 13,255 | 0.8555 | 0.9738 | 0.8303 | 11,302 |  |
| `sensitive_pii_social_security_number` ★ | 6,748 | 0.8626 | 0.9272 | 0.8478 | 6,170 |  |
| `sensitive_pci_individual_taxpayer_identification_number_itin` ★ | 7,672 | 0.8719 | 0.9717 | 0.8501 | 6,712 |  |
| `sensitive_pii_area_code` | 15,314 | 0.8830 | 0.9726 | 0.8631 | 13,589 |  |
| `sensitive_pii_address` ★ | 15,267 | 0.9097 | 0.6974 | 0.9847 | 21,557 |  |

*fusion-12k (arm C)*

| tag | n | F2 | precision | recall | predicted | state |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sensitive_pii_country_of_origin` | 3 | 0.0000 | 0.0000 | 0.0000 | 15 |  |
| `sensitive_pii_country_of_residence` | 3 | 0.0000 | 0.0000 | 0.0000 | 10 |  |
| `sensitive_pii_nationality` | 3 | 0.0000 | 0.0000 | 0.0000 | 7 |  |
| `sensitive_pii_username` | 2 | 0.0000 | 0.0000 | 0.0000 | 33 |  |
| `sensitive_pci_iban` ★ | 1 | 0.0002 | 0.0000 | 1.0000 | 24,275 |  |
| `sensitive_pci_bank_account_number` ★ | 1 | 0.0002 | 0.0000 | 1.0000 | 23,673 |  |
| `sensitive_pii_ipv4` | 2 | 0.2000 | 0.0588 | 0.5000 | 17 |  |
| `sensitive_pii_social_security_number` ★ | 6,748 | 0.5146 | 0.1749 | 1.0000 | 38,579 |  |
| `sensitive_pci_individual_taxpayer_identification_number_itin` ★ | 7,672 | 0.5568 | 0.2008 | 1.0000 | 38,209 |  |
| `sensitive_pii_driver_s_license_number` ★ | 8,538 | 0.5845 | 0.2196 | 0.9999 | 38,870 |  |
| `sensitive_pci_financial_instrument_global_identifier_figi` | 8,538 | 0.5931 | 0.3022 | 0.7812 | 22,075 |  |
| `sensitive_pci_credit_card_number` ★ | 8,908 | 0.6005 | 0.2311 | 1.0000 | 38,542 |  |
| `sensitive_pii_passport_number` ★ | 13,763 | 0.7337 | 0.3554 | 0.9997 | 38,710 |  |
| `sensitive_pii_visa_number` ★ | 13,763 | 0.7339 | 0.3558 | 0.9995 | 38,664 |  |
| `sensitive_pii_military_identification_number` ★ | 13,763 | 0.7339 | 0.3558 | 0.9996 | 38,667 |  |
| `sensitive_pii_last_four_us_ssn_digits` | 6,748 | 0.7475 | 0.6718 | 0.7691 | 7,726 |  |
| `sensitive_pii_immigration_and_citizenship_status` | 13,763 | 0.7548 | 0.4764 | 0.8840 | 25,540 |  |
| `sensitive_pii_address` ★ | 15,267 | 0.7633 | 0.3921 | 1.0000 | 38,933 |  |
| `sensitive_pii_age` | 13,255 | 0.7954 | 0.5485 | 0.8963 | 21,658 |  |
| `sensitive_pii_area_code` | 15,314 | 0.8002 | 0.7562 | 0.8121 | 16,446 |  |

★ = priority tag. **unlearned** = never predicted once; *silent* = recognised but almost never emitted.

## Decision — headline

Selected: **arm-C**

| Arm | Verdict | Hard constraints failed | Scoped detail |
| --- | --- | ---: | --- |
| arm-A | **FEASIBLE** | 0 of 2 | — |
| arm-B | blocked | 1 of 2 | 25/55 measurable scope(s) passed of 128 |
| arm-C | **WINNER** | 0 of 2 | — |

- **arm-A** — cleared every hard constraint.
- **arm-B** — 1 hard constraint(s) failed:
    - *priority tag recall >= 0.90 (conclusive)* — 25/55 measurable scope(s) passed of 128
        - `sensitive_pii_password@10360_betterdataai_ner_silver_eval_10.36k` — 0.2400 (point 0.3600, n=50)
        - `sensitive_phi_medical_record_number_mrn@10360_betterdataai_ner_silver_eval_10.36k` — 0.3510 (point 0.4305, n=151)
        - `sensitive_pii_address@6589_govdocs2-dualjudge-eval20-3.53k` — 0.5252 (point 0.5589, n=891)
        - `sensitive_pii_full_name@6589_govdocs2-dualjudge-eval20-3.53k` — 0.5493 (point 0.5670, n=3215)
        - `sensitive_pii_address@4000_datax-dualjudge-evalset-1.32k` — 0.5914 (point 0.6429, n=350)
        - `sensitive_pii_address@10360_betterdataai_ner_silver_eval_10.36k` — 0.6063 (point 0.6614, n=254)
        - `sensitive_pii_personal_identification_number_pin@10626_ai4privacy_pii_masking_eval_10.63k` — 0.6069 (point 0.6828, n=145)
        - `sensitive_pii_full_name@10626_ai4privacy_pii_masking_eval_10.63k` — 0.6755 (point 0.6870, n=5470)
        - `sensitive_pii_driver_s_license_number@38937_openpii_pii_eval_38.94k` — 0.6897 (point 0.6993, n=8538)
        - `sensitive_pii_social_security_number@10626_ai4privacy_pii_masking_eval_10.63k` — 0.7054 (point 0.7242, n=2400)
        - …and 20 more
- **arm-C** — cleared every hard constraint.

## Decision — precision_view

Selected: **none — no feasible arm**

| Arm | Verdict | Hard constraints failed | Scoped detail |
| --- | --- | ---: | --- |
| arm-A | blocked | 2 of 5 | 0/3 measurable scope(s) passed of 3; 0/3 measurable scope(s) passed of 3 |
| arm-B | blocked | 3 of 5 | 1/3 measurable scope(s) passed of 3; 1/3 measurable scope(s) passed of 3; 43/55 measurable scope(s) passed of 128 |
| arm-C | blocked | 2 of 5 | 0/3 measurable scope(s) passed of 3; 0/3 measurable scope(s) passed of 3 |

- **arm-A** — 2 hard constraint(s) failed:
    - *document-level precision >= 0.90 (conclusive)* — 0/3 measurable scope(s) passed of 3
        - `doc@4000_datax-dualjudge-evalset-1.32k` — 0.4615 (point 0.4783, n=4000)
        - `doc@6589_govdocs2-dualjudge-eval20-3.53k` — 0.5204 (point 0.5323, n=6545)
        - `doc@30000_pii2_eval_25.15k` — 0.8344 (point 0.8384, n=29998)
    - *document-level specificity >= 0.85 (conclusive)* — 0/3 measurable scope(s) passed of 3
        - `doc@30000_pii2_eval_25.15k` — 0.0000 (point 0.0004, n=4851)
        - `doc@4000_datax-dualjudge-evalset-1.32k` — 0.0000 (point 0.0000, n=2087)
        - `doc@6589_govdocs2-dualjudge-eval20-3.53k` — 0.0000 (point 0.0013, n=3065)
- **arm-B** — 3 hard constraint(s) failed:
    - *document-level precision >= 0.90 (conclusive)* — 1/3 measurable scope(s) passed of 3
        - `doc@4000_datax-dualjudge-evalset-1.32k` — 0.8102 (point 0.8284, n=1568)
        - `doc@6589_govdocs2-dualjudge-eval20-3.53k` — 0.8474 (point 0.8601, n=2631)
    - *document-level recall >= 0.85 (conclusive)* — 1/3 measurable scope(s) passed of 3
        - `doc@6589_govdocs2-dualjudge-eval20-3.53k` — 0.6330 (point 0.6495, n=3484)
        - `doc@4000_datax-dualjudge-evalset-1.32k` — 0.6576 (point 0.6790, n=1913)
    - *priority tag recall >= 0.75 (conclusive)* — 43/55 measurable scope(s) passed of 128
        - `sensitive_pii_password@10360_betterdataai_ner_silver_eval_10.36k` — 0.2400 (point 0.3600, n=50)
        - `sensitive_phi_medical_record_number_mrn@10360_betterdataai_ner_silver_eval_10.36k` — 0.3510 (point 0.4305, n=151)
        - `sensitive_pii_address@6589_govdocs2-dualjudge-eval20-3.53k` — 0.5252 (point 0.5589, n=891)
        - `sensitive_pii_full_name@6589_govdocs2-dualjudge-eval20-3.53k` — 0.5493 (point 0.5670, n=3215)
        - `sensitive_pii_address@4000_datax-dualjudge-evalset-1.32k` — 0.5914 (point 0.6429, n=350)
        - `sensitive_pii_address@10360_betterdataai_ner_silver_eval_10.36k` — 0.6063 (point 0.6614, n=254)
        - `sensitive_pii_personal_identification_number_pin@10626_ai4privacy_pii_masking_eval_10.63k` — 0.6069 (point 0.6828, n=145)
        - `sensitive_pii_full_name@10626_ai4privacy_pii_masking_eval_10.63k` — 0.6755 (point 0.6870, n=5470)
        - `sensitive_pii_driver_s_license_number@38937_openpii_pii_eval_38.94k` — 0.6897 (point 0.6993, n=8538)
        - `sensitive_pii_social_security_number@10626_ai4privacy_pii_masking_eval_10.63k` — 0.7054 (point 0.7242, n=2400)
        - …and 2 more
- **arm-C** — 2 hard constraint(s) failed:
    - *document-level precision >= 0.90 (conclusive)* — 0/3 measurable scope(s) passed of 3
        - `doc@4000_datax-dualjudge-evalset-1.32k` — 0.4615 (point 0.4783, n=4000)
        - `doc@6589_govdocs2-dualjudge-eval20-3.53k` — 0.5204 (point 0.5324, n=6544)
        - `doc@30000_pii2_eval_25.15k` — 0.8344 (point 0.8384, n=29997)
    - *document-level specificity >= 0.85 (conclusive)* — 0/3 measurable scope(s) passed of 3
        - `doc@30000_pii2_eval_25.15k` — 0.0000 (point 0.0006, n=4851)
        - `doc@4000_datax-dualjudge-evalset-1.32k` — 0.0000 (point 0.0000, n=2087)
        - `doc@6589_govdocs2-dualjudge-eval20-3.53k` — 0.0003 (point 0.0016, n=3065)

