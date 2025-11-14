#!/bin/bash

echo "提交开始"
git add .
git commit -m "update"
git push
echo "提交结束"
read -p "按任意键继续..."
