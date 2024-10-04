#!/bin/bash

# 定义要使用的条件和模型
select_conditions=("UDDS" "FUDS" "US06")
models=("SPP" "Conv" "Vit")

# 循环遍历每种组合
for condition in "${select_conditions[@]}"; do
    for model in "${models[@]}"; do
        # 运行 Python 脚本
        python main.py --select_condition "$condition" --model "$model" --save_model --save_res --counts 1004
    done
done
