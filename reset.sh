#!/bin/bash

read -p "确定重置吗(Y/[N])? " areyousure

# 检查输入是否为Y或y（不区分大小写）
if [[ "$areyousure" =~ ^[Yy]$ ]]; then
    echo "重置开始"
    git reset --hard HEAD~1
    echo "重置结束"
fi

read -p "按任意键继续..."
