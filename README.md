# LAnguage-Modeling-Is-All-You-Need-for-Lifelong-Language-Learning
Most research on lifelong learning (LLL) applies to images or games, but not
language.
We present LAMAL, a simple yet effective method for LLL based on language
modeling.
LAMAL replays pseudo-samples of previous tasks while requiring no extra memory
or model capacity.
Specifically, LAMAL is a language model that simultaneously learns to solve the
task and generate training samples.
When the model is trained for a new task, it generates pseudo-samples of
previous tasks for training alongside data for the new task.
The results show that LAMAL prevents catastrophic forgetting without any sign of
intransigence and can perform up to five very different language tasks
sequentially with only one model. 
Overall, LAMAL outperforms previous methods by a considerable margin and is only
2--3\% worse than multitasking, which is usually considered the LLL upper bound.

## Dataset
We first ran the code in https://github.com/salesforce/decaNLP to get the dataset, and then converted them into Squad-like format.

## Dependencies
- Ubuntu >= 16.04
- This code only supports the following GPUs:
  - NVIDIA Geforce RTX 2080TI 
  - NVIDIA TESLA V100
- python3
- cuda 10.1
- python packages are listed in `requirements.txt`

## Setup
1. Create the following two directories in wherever you want. (you can name the directories arbitrarily):
    - `data directory`: Where the dataset will be load by the model.
    - `model directory`: The place for the model to dump its outputs.
2. Download the dataset: . After decompression, move all the files in the decompressed directory into `data directory`.
3. Make a copy of `env.example` and save it as `env`. In `env`, set the value of DATA_DIR as `data directory` and set the value of  MODEL_ROOT_DIR as `model directory`.
4. Before training or testing, load DATA_DIR and MODEL_ROOT_DIR variables into shell environment by the following command:
   ```bash 
   source env
   ```

## Training and Testing

`train.sh` and `test.sh` are the entrance for training and testing. Main options for them include:

| Options        | Description   |
| -------------  | ------------- |
| seq_train_type | The mode to deal with a sequence of tasks. Mode include: lll\|finetune\|multitask\|mas\|ewc. "lll" is the default value corresponding our proposed method. The others are the methods for comparing with our proposal. |
| tasks          | A sequence of tasks we want to train by seq_train_type. Leave a space between tasks after the `--tasks` tag. Tasks are the keys in TASK_DICT variable in `settings.py` |
| model_name     | The language model we want to use. The default is `gpt2`. Options include gpt2\|openai-gpt, |
| gen_lm_sample_percentage | This tag only works with `--seq_train_type lll`. The percentage of the size of the dataset will be generated as pseudo samples for our proposed method. |
| lm_lambda      | Lambda value for the loss function. |
| max_n_epochs   | Maximum epoch value for all tasks. |
| min_batch_size | Minimum batch size for all tasks. |
| min_n_steps    | Minimum step for optimizing the model for all tasks. |
| n_train_epochs | Epochs for training for all tasks. |
| n_gpu          | Number of gpu to be used. |
| reg_lambda     | Lambda value for mas and ewc. |
| top_k_lm       | Top k sampling for the language model. |
| top_k_qa       | Top k sampling for the qa model. |
| train_batch_size | Batch size for all tasks. The default is 0. Once the value equals to 0, The batch size will be decided dynamically based on the memory usage of the gpu. |

### Training 

#### Example:

If you want to train sst, srl and woz.en sequentially by our proposed method, run:
```bash
./train.sh --seq_train_type lll --tasks sst srl woz.en
```

#### Outputs:

If assigning multitask to `--seq_train_type` tag, the model will be dumped in `$MODEL_ROOT_DIR / model_name / seq_train_type /TASK1_TASK2_...` directory. Otherwise, it will be in `$MODEL_ROOT_DIR / model_name / seq_train_type / TASK1_TASK2_... / TASK1`, `$MODEL_ROOT_DIR / model_name / seq_train_type / TASK1_TASK2_... / TASK2`, ... directories. 

For example, if you run:
```bash
./train.sh --seq_train_type multitask --tasks sst srl woz.en
```
Then the model will be dumped in: `$MODEL_ROOT_DIR/gpt2/multitask/squad1_wikisql_sst_srl_woz.en`.

If you run:
```bash
./train.sh --seq_train_type lll --model_name openai-gpt --gen_lm_sample_percentage 0.2 --tasks sst srl woz.en
```
Then the models will be dumped in the following directories: `$MODEL_ROOT_DIR/openai-gpt/lll/sst_srl_woz.en_0.2/sst`, `$MODEL_ROOT_DIR/openai-gpt/lll/sst_srl_woz.en_0.2/srl`, `$MODEL_ROOT_DIR/openai-gpt/lll/sst_srl_woz.en_0.2/woz.en`.


### Testing

#### Example:

This example test the model trained on sst, srl and woz.en by finetune method.
```bash
./test.sh --seq_train_type finetune --tasks sst srl woz.en
```

#### Outputs:
After running testing program, the metrics: `metrics.json` will be dumped in the same directory of Training's outputs.

## Acknowledgements:

