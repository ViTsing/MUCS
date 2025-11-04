import os
import zlib
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
from scipy import stats
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 命令行示例：python com_det_sco_new.py --tar_mod=llama-3.1-8b --data=tulu3-00000-of-00006

def get_arg():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tar_mod", type=str, default="pythia-2.8b") # 目标模型
    parser.add_argument("--data", type=str, default="WikiMIA_128") # 被测数据集
    parser.add_argument("--ref_dat",type=str,default="c4") # 参考语料库
    parser.add_argument("--a", type=float, default="0.01") # 超参数
    arg = parser.parse_args()
    return arg


def cal_met(pro_dis, fre_dis_smo, a):
    '''
    计算多种检测方法的预测分数，用于判断文本是否属于预训练数据
    '''
    es = []

    # 遍历样本：
    for i, t in enumerate(tqdm(pro_dis)):

        e = {}

        # 计算DC-PDD方法
        probs = np.exp(t["tar_prob"]) # (0,1)
        input_ids = np.array(t["input_ids"])

        if len(input_ids)==0:
            continue
        
        # (3) 和第一次出现保持一致
        indexes = []
        current_ids = []
        for i, input_id in enumerate(input_ids):
            if input_id < len(fre_dis_smo): # 防止有溢出的ID
                indexes.append(i)
                if input_id not in current_ids:
                    current_ids.append(input_id)
                else:
                    index = current_ids.index(input_id)
                    probs[i] = probs[index]

        for i in range(1,len(probs)):
            if probs[i]-probs[i-1]>=0.725441:
                probs[i] = probs[i-1]

        x_pro = probs[indexes] # 每个token的首次出现概率
        x_fre = fre_dis_smo[input_ids[indexes]] # 每个token的参考频率

        if len(x_pro)!=len(x_fre):
            print("Wrong!!!!!")

        ce = - x_pro * np.log(x_fre)
        ce[ce > a] = a

        e['text'] = t['text']
        e['score'] = np.mean(ce)
        e['idx'] = t['idx']

        # # 预测标签
        # if e['score'] <= 0.00888792:
        #     e['is_pretrained'] = 0
        # else:
        #     e['is_pretrained'] = 1

        es.append(e)  # DC-PDD方法的预测分数

    return es


if __name__ == "__main__":
    args = get_arg()

    # 加载概率分布
    pro_dis_dir = "output/pro_dis"
    pro_dis_pat = os.path.join(pro_dis_dir, args.data, f"{args.tar_mod}.pkl")
    with open(pro_dis_pat, "rb") as f:
        pro_dis = pkl.load(f)  # com_pro_dis.py的生成结果

    # 加载频率分布
    fre_dis_dir = "output/fre_dis"
    fre_dis_pat = os.path.join(fre_dis_dir, args.ref_dat, f"{args.tar_mod}.pkl")
    with open(fre_dis_pat, "rb") as f:
        fre_dis = pkl.load(f)  # com_fre_dis.py的生成结果

    # 平滑频率分布
    fre_dis_npy = np.array(fre_dis) # count
    fre_dis_smo = (fre_dis_npy + 1) / (sum(fre_dis_npy) + len(fre_dis_npy)) # (count+1) / (N + V)

    # 计算预测分数
    det_sco = cal_met(pro_dis, fre_dis_smo, args.a)

    # 创建输出目录
    out_dir = "output/det_sco"
    out_path = os.path.join(out_dir, args.data)
    Path(out_path).mkdir(parents=True, exist_ok=True)

    with open(f"{out_path}/{args.tar_mod}.pkl", "wb") as f:
        pkl.dump(det_sco, f)


# examples=[e1,e2,e3,...]
# e1 = {
#       ‘text':...,
#       'score':..., # DC-PDD的检测分数
#       'idx':...,
#    
# }