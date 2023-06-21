import os
basic_info="""
# Basic Information:

- Title: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (低成本硬件学习细粒度双手操作)
- Authors: Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn
- Affiliation: Stanford University (斯坦福大学)
- Keywords: fine manipulation, low-cost hardware, imitation learning, teleoperation, action chunking (细粒度操作，低成本硬件，模仿学习，远程操作，动作分块)
- URLs: [Paper](https://arxiv.org/abs/2304.13705v1), [GitHub](https://tonyzhaozh.github.io/aloha)
"""


res = basic_info.split("\n- ")[1].replace("Title: ", '')
print(res)

print(len("Introduction of an exclusive, highly linear, and matrix-effectless analytical method based on dispersive micro solid phase extraction using MIL-88B(Fe) followed by dispersive liquid–liquid microextraction specialized for the analysis of pesticides in celery and tomato juices without dilution"))
# print(os.path.getsize('../uploads/3047b38215263278f07178419489a887.pdf')/1024,'KB')