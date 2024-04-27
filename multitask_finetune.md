## Finetune from the pretrained multi-task model
Now we have a multi-task model `model_18heads.pt` with 18 fitting heads, with different branches such as `Domains_Alloy`, `Domains_Anode`, `Domains_Cluster`, `Domains_Drug`, etc. 

The content of `input.json` is different from single-task training. Includes a `model/shared_dict <model/shared_dict>`shared by all models, such as `dpa2_descriptor`, and multiple model definitions `model/model_dict/model_key <model/model_dict/model_key>`instead of a single model definition `model <model>`.

For example, I want to run finetune on a new system while continuing to train on the relevant data set of the pretrain model such as `Domains_Alloy` to prevent overfitting. We can define model_dict as follows:
```json
"model_dict": {
            "Domains_Alloy": {
                "type_map": "type_map_all",
                "descriptor": "dpa2_descriptor",
                "fitting_net": {
                    "neuron": [
                        240,
                        240,
                        240
                    ],
                    "activation_function": "tanh",
                    "resnet_dt": true,
                    "seed": 1,
                    "_comment": " that's all"
                }
            },
            "new_system": {
                "type_map": "type_map_all",
                "descriptor": "dpa2_descriptor",
                "fitting_net": {
                    "neuron": [
                        240,
                        240,
                        240
                    ],
                    "activation_function": "tanh",
                    "resnet_dt": true,
                    "seed": 1,
                    "_comment": " that's all"
                }
            }
        }
```
Correspondingly, we need to define `loss_dict <loss_dict>` and `training/data_dict <training/data_dict>` for each task, and control the weights between different heads through `training/model_prob <training/model_prob>`:
```json
    "loss_dict": {
        "Domains_Alloy": {
            "type": "ener",
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_f": 1000,
            "limit_pref_f": 1,
            "start_pref_v": 0,
            "limit_pref_v": 0
        },
        "new_system": {
            "type": "ener",
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_f": 1000,
            "limit_pref_f": 1,
            "start_pref_v": 0,
            "limit_pref_v": 0
        }
    },
```
```json
    "training": {
        "model_prob": {
            "Domains_Alloy": 2.0,
            "new_system": 3.0
        },
        "data_dict": {
            "Domains_Alloy": {
                "training_data": {
                    "systems": [
                        "/home/data/Domains/Alloy/train/9",
                        "/home/data/Domains/Alloy/train/10",
                        "/home/data/Domains/Alloy/train/11"
                    ],
                    "batch_size": "auto",
                    "_comment": "that's all"
                },
                "validation_data": {
                    "systems": [
                        "/home/data/Domains/Alloy/val/9",
                        "/home/data/Domains/Alloy/val/10",
                        "/home/data/Domains/Alloy/val/11"
                    ],
                    "batch_size": "auto",
                    "_comment": "that's all"
                }
            },
            "new_system": {
                "training_data": {
                    "systems": [
                        "/home/data/new_system/train/9",
                        "/home/data/new_system/train/10",
                        "/home/data/new_system/train/11"
                    ],
                    "batch_size": "auto",
                    "_comment": "that's all"
                },
                "validation_data": {
                    "systems": [
                        "/home/data/new_system/val/9",
                        "/home/data/new_system/val/10",
                        "/home/data/new_system/val/11"
                    ],
                    "batch_size": "auto",
                    "_comment": "that's all"
                }
            },
        }
    }
```
Once we have finished setting up `input.json`, we are ready to train:
```bash
$ dp --pt train input.json --finetune model_18heads.pt
```