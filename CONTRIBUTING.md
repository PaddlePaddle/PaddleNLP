**简体中文**🀄 | [English🌎](.github/CONTRIBUTING_en.md)

# Contributing to PaddleNLP

我们非常欢迎并希望您对`PaddleNLP`做出开源贡献。在您开始提交您的贡献之前，请先行签署[PaddlePaddle 贡献者许可协议](https://cla-assistant.io/PaddlePaddle/PaddleNLP)。
本文接下来将介绍我们的开发与贡献流程：

## 贡献方式

我们欢迎不同的向`PaddleNLP`做出贡献的方式，例如：

- 修复已知的 Issue
- 提交新的 Issue，例如提出功能需求或者 bug 报告
- 实现新的模型结构

如果您不知道从哪里开始，请查看 Issues 板块中的`Good First Issue`标签。它为您提供一个对初学者友好的已知 Issue 列表，可以降低贡献的门槛，帮助您开始为开源做出贡献。您只需在您想处理的 Issue 中告知我们您想负责此 Issue 即可。

## 开发流程

PaddleNLP 使用 [Git 分支模型](http://nvie.com/posts/a-successful-git-branching-model/)。对于常见的开源贡献，我们有以下的贡献流程：

### 1. Fork

   因为 PaddleNLP 的开发社区一直在发展，如果每位贡献者都直接向官方 Repo 提交 commit 将会难以管理。因此，请从您的分支中提交 Pull Requests。建议您通过 GitHub 的[“Fork”按钮](https://help.github.com/articles/fork-a-repo/)来创建您的 Fork 分支。

### 2. Clone

   请运行一下命令将您的分支 clone 到本地

   ```bash
   git clone https://github.com/<your-github-account>/PaddleNLP
   cd PaddleNLP
   ```

### 3. 创建本地开发分支

   对于添加新功能或修复错误等日常工作，请在开发前创建您的本地开发分支：

   ```bash
   git checkout -b my-cool-feature
   ```

### 4. 配置开发环境

   在开始编码之前，您需要设置开发环境。我们强烈建议您在虚拟环境中进行所有开发，例如[venv](https://docs.python.org/3/library/venv.html)或[conda](https://docs.conda.io/en/latest/)。
   请您设置并激活虚拟环境后，运行以下命令：

   ```bash
   make install
   ```

   这将设置 `PaddleNLP` 的所有依赖以及 [`pre-commit`](http://pre-commit.com/) 工具。

   如果您需要开发 `examples` 或 `applications` 模块并加载 `PaddleNLP`，请确保以可编辑模式（`-e`）安装 `PaddleNLP`。
   如果在虚拟环境中已经安装 `PaddleNLP` ，请使用 `pip uninstall paddlenlp` 将其删除，然后以可编辑模式重新安装它
   `pip install -e .`

### 5. 开发

   当您开发时，请确保您新增的代码会被单元测试所覆盖。我们所有的单元测试都可以在 `tests` 目录下找到。
   您可以修改现有单元测试以覆盖新功能，也可以从头开始创建新测试。
   当您完成代码时，您应该确保相关的单元测试可以通过。您可以像这样运行受更改影响的测试：

   ```bash
   pytest tests/<test_to_run>.py
   ```

### 6. Commit

   我们使用 [`pre-commit`](http://pre-commit.com/)工具（包括[black](https://black.readthedocs.io/en/stable/)、[isort](https:/ /pycqa.github.io/isort/) 和
   [flake8](https://flake8.pycqa.org/en/latest/)）来检查每次提交中的代码和文档的风格。当你运行 `git commit` 时，你会看到
   类似于以下内容：

   ```text
    ➜  (my-virtual-env) git commit -m "commiting my cool feature"
    black....................................................................Passed
    isort....................................................................Passed
    flake8...................................................................Passed
    check for merge conflicts................................................Passed
    check for broken symlinks............................(no files to check)Skipped
    detect private key.......................................................Passed
    fix end of files.....................................(no files to check)Skipped
    trim trailing whitespace.............................(no files to check)Skipped
    CRLF end-lines checker...............................(no files to check)Skipped
    CRLF end-lines remover...............................(no files to check)Skipped
    No-tabs checker......................................(no files to check)Skipped
    Tabs remover.........................................(no files to check)Skipped
    copyright_checker........................................................Passed
   ```

   但大多数时候事情并没有那么顺利。当您的代码或文档不符合标准时，`pre-commit` 检查将失败。

   ```text
    ➜  (my-virtual-env) git commit -m "commiting my cool feature"
    black....................................................................Passed
    isort....................................................................Failed
    - hook id: isort
    - files were modified by this hook

    Fixing examples/information_extraction/waybill_ie/run_ernie_crf.py

    flake8...................................................................Passed
    check for merge conflicts................................................Passed
    check for broken symlinks............................(no files to check)Skipped
    detect private key.......................................................Passed
    fix end of files.....................................(no files to check)Skipped
    trim trailing whitespace.............................(no files to check)Skipped
    CRLF end-lines checker...............................(no files to check)Skipped
    CRLF end-lines remover...............................(no files to check)Skipped
    No-tabs checker......................................(no files to check)Skipped
    Tabs remover.........................................(no files to check)Skipped
    copyright_checker........................................................Passed
   ```

   我们的工具将自动修复大部分样式错误，但是有些错误需要手动解决。幸运的是，错误信息一般通俗易懂，很容易修复。
   解决错误后，您可以再次运行 `git add <files>` 和 `git commit`，这将再次触发 pre-commit 。
   一旦 pre-commit 检查通过，您就可以推送代码了。

   [Google](https://google.com/) 或 [StackOverflow](https://stackoverflow.com/) 是帮助您了解代码风格错误的好工具。
   如果您仍然无法弄清楚，请不要担心。您可以使用 `git commit -m "style error" --no-verify` 提交，我们很乐意在您创建 Pull Request 后帮助您。

### 7. git pull 与代码冲突

   有经验的 Git 用户经常从官方 Repo 中 git pull。因为这样子他们会及早注意到与其他人的代码冲突，并且让代码冲突更容易解决

   ```bash
   git remote add upstream https://github.com/PaddlePaddle/PaddleNLP
   git pull upstream develop
   ```

### 8. git push 与提交 Pull Request

   您可以将您的本地开发分支中的工作 push 到您的 fork 的分支中：

   ```bash
   git push origin my-cool-stuff
   ```

   git push 之后，您可以提交 Pull Request，请求[官方 repo](https://github.com/PaddlePaddle/PaddleNLP) 采纳您的开发工作。请您依照[这些步骤](https://help.github.com/articles/creating-a-pull-request/)创建 Pull Request。

### 9. 删除已经合入的本地和远程分支

   为了保持您本地的工作区和 fork 分支的干净整洁，建议您在 Pull Request 合入之后删除本地的残余分支：

   ```bash
   git push origin my-cool-stuff
   git checkout develop
   git pull upstream develop
   git branch -d my-cool-stuff
   ```

## 代码 Review

- 在您的 Pull Request 能够顺利通过本地测试以及 CI 的情况下，您可以在 Pull Request 中 @ 相关的 Reviewer，提醒他们尽快对您的 Pull Request 进行 Review。

- 请处理 Reviewer 的每一条评论。如果您已按照评论修改，请回复“完成”；否则，可以在评论下展开讨论。

- 如果您不希望您的 Reviewer 被电子邮件通知淹没，您可以[批量回复](https://help.github.com/articles/reviewing-proposed-changes-in-a-pull-request/)。
