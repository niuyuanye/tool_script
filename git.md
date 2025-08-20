#### 切换分支
```
git checkout remotes/origin/feature/李石-魔方智盒管理平台
git switch remotes/origin/feature/李石-魔方智盒管理平台

```


#### 新建分支并推送远端
```
# 1. 基于 main 创建新分支
git switch -c feature/nyy
git switch -c feature/牛原野-魔方智盒算法
git switch -c feature/牛原野-可视化训练平台
# 2. 开发新功能...
echo "新代码" >> file.js
git add .
git commit -m "添加nyy用户资料页"

# 3. 推送到远程并建立关联
git push -u origin feature/nyy
git push -u origin feature/牛原野-魔方智盒算法
git push -u origin feature/牛原野-可视化训练平台
# 4. 后续继续开发...
git add .
git commit -m "完善个人资料编辑功能"
git push  # 简化推送
```
#### 修该本地分支名称

```
# 切换到要修改的分支
git checkout old-branch-name
# 直接重命名当前分支
git branch -m new-branch-name
# 使用 switch 命令（Git 2.23+）
git switch -c new-branch-name
git switch -c feature/牛原野-魔方智盒算法

```

#### 修改远程分支名称

```
步骤 1：删除旧远程分支
git push origin --delete old-branch-name
git push origin --delete feature/牛原野-魔方智盒算法
步骤 2：推送新分支到远程
git push origin -u new-branch-name
git push origin -u feature/牛原野-魔方智盒算法
```

#### 如果远程分支已被其他人使用（协作场景）

```
# 1. 通知团队成员暂停在旧分支的工作
# 2. 创建新分支
git checkout -b new-feature

# 3. 推送新分支
git push -u origin new-feature

# 4. 删除旧分支（确保所有变更已合并）
git push origin --delete old-feature

# 5. 更新团队成员：
#    - 删除本地旧分支：git branch -D old-feature
#    - 重新拉取新分支：git checkout new-feature
```



#### 高级技巧：保留提交历史

```
# 1. 基于旧分支创建新分支
git checkout -b new-feature old-feature

git checkout -b  feature/牛原野-魔方智盒算法 feature/nyy

# 2. 推送新分支
git push -u origin new-feature
git push -u origin feature/牛原野-魔方智盒算法

# 3. 删除旧分支（本地和远程）
git branch -d feature/nyy
git push origin --delete feature/nyy
```



#### 合并分支

```
切换到目标分支
首先，确保你的本地仓库是最新的，并且切换到develop分支。你可以使用以下命令来更新你的本地仓库并切换到develop分支：
git checkout develop
git pull origin develop

合并分支
你需要将你的分支合并到develop分支。假设你的分支名称为feature-branch，你可以使用以下命令来合并：
git merge feature-branch

git checkout develop
git merge feature/牛原野-魔方智盒算法
git push origin develop

在合并过程中，如果出现冲突，Git会提示你解决这些冲突。你需要手动编辑引起冲突的文件，选择性地保留更改，然后添加并提交这些更改：
git add <文件名>
git commit

推送更改到远程仓库
git push origin develop
```



#### 强制使用远程分支覆盖本地分支

如果您希望完全使用远程分支的代码覆盖本地分支（注意：此操作会丢弃本地所有未提交的更改），可以执行：

```
git reset --hard 
git pull origin feature/牛原野-可视化训练平台


git fetch --all
git reset --hard feature/牛原野-可视化训练平台
git pull origin feature/牛原野-可视化训练平台 --force
```

