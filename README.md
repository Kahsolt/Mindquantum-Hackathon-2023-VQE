# MindSpore Quantum Compuation Hackathon 2023

    量子信息技术与应用创新大赛 —— 2023 MindSpore 量子计算黑客松全国大赛

----

比赛页面: [https://competition.huaweicloud.com/information/1000041884/introduction](https://competition.huaweicloud.com/information/1000041884/introduction)

比赛分为三个阶段：

- 热身赛 stage0
- 初赛 stage1
- 决赛 stage2


### install

⚠ The dependent package `pyscf` does NOT support native Windows, so you must run code under Linux :(

- create venv
  - `conda create -n mq python==3.9`
  - `conda activate mq`
- install mindspore
  - follow `https://www.mindspore.cn/install`
  - install combination `2.0.0 + CPU + Windows-x64 + Python 3.9 + Pip`
  - fix PIL version compatible error: `pip install Pillow==9.5.0`
  - test installation `python -c "import mindspore;mindspore.run_check()"`
- `pip install mindquantum`
- `pip install -r requirements.txt`


### references

- MindSpore: [https://www.mindspore.cn](https://www.mindspore.cn)
  - install: [https://gitee.com/mindspore/docs/blob/r2.0/install/mindspore_cpu_win_install_pip.md](https://gitee.com/mindspore/docs/blob/r2.0/install/mindspore_cpu_win_install_pip.md)
  - tutorial: [https://www.mindspore.cn/tutorials/zh-CN/r2.0/index.html](https://www.mindspore.cn/tutorials/zh-CN/r2.0/index.html)
- MindQuantum: [https://gitee.com/mindspore/mindquantum](https://gitee.com/mindspore/mindquantum)
  - doc: [https://www.mindspore.cn/mindquantum/docs/zh-CN/master/index.html](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/index.html)
- QuPack:
  - tutorial: [https://hiq.huaweicloud.com/document/QAOA](https://hiq.huaweicloud.com/document/QAOA)
  - doc: [https://hiq.huaweicloud.com/document/qupack-0.1.1/index.html](https://hiq.huaweicloud.com/document/qupack-0.1.1/index.html)

- pyChemiQ
  - repo: [https://github.com/OriginQ/pyChemiQ](https://github.com/OriginQ/pyChemiQ)
  - doc: [https://pychemiq-tutorial.readthedocs.io/en/latest/index.html](https://pychemiq-tutorial.readthedocs.io/en/latest/index.html)

----

by Armit
2023/07/07 
