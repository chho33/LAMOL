from utils_lll import create_dataloader, QADataset
from settings_lll import args,  TASK_DICT
import pickle as pkl
import re
import os
import json
from multiprocessing import Pool


def serialize_data(redo=True): 
    print("serializing data ...")
    for t in ["train", "eval", "test"]:
        for task in TASK_DICT.keys():
            data_path = TASK_DICT[task][t]
            pkl_path = re.sub("json","pkl", data_path)
            if os.path.exists(pkl_path) and not redo:
                continue
            dataset = QADataset(data_path, t)
            with open(pkl_path, "wb") as f:
                pkl.dump(dataset,f)
    print("data serialized!")
            

def dump_data_attrs(task):
    attrs = {task:{"train":{}, "eval":{}, "test":{}}}
    for t in ["train", "eval", "test"]:
        print(task,t)
        data_path = TASK_DICT[task][t]
        pkl_path = re.sub("json","pkl", data_path)
        with open(pkl_path, "rb") as f:
            dataset = pkl.load(f)
        attrs[task][t] = {"data_size": len(dataset),
                          "max_a_len": dataset.max_a_len,
        }
    return attrs 


def parallel_dump_data_attrs(tasks=TASK_DICT.keys()):
    print("creating data_attrs.json ...")
    attr_dict = {}
    with Pool(args.n_workers) as pool:
        attrs = pool.map(dump_data_attrs, tasks)
    for a in attrs: 
        attr_dict.update(a)
    with open("data_attrs.json","w") as f:
        json.dump(attr_dict,f)
    print("data_attrs.json created!")


serialize_data()
parallel_dump_data_attrs()
