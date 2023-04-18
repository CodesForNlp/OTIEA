import matplotlib
import matplotlib.pyplot as plt
import numpy as np
###带标准差的柱状图
plt.rcParams['font.sans-serif']=['SimHei'] # 解决中文乱码
font1 = {
'family' : 'Times New Roman',
# 'weight' : 'bold',
'size'   : 15,
}
font2 = {
'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 9,
}

# #与最好的几个结果比较
# #柱状图
# name_list = ['H@1', 'H@10', 'MRR']
# # #zh_en
# # num_list=[0.798, 0.933, 0.847]    #RAGA-l
# # num_list1=[0.800, 0.934, 0.849]   #TTEA-base
# # num_list2=[0.865, 0.964, 0.902]   #TTEA-semi
# # num_list3=[0.763, 0.914, 0.835]   #SHEA
# # num_list4=[0.736, 0.873, 0.786]   #KAGNN
# # #ja_en
# # num_list=[0.829, 0.950, 0.875]    #RAGA-l
# # num_list1=[0.837, 0.953, 0.880]   #TTEA-base
# # num_list2=[0.890, 0.972, 0.922]   #TTEA-semi
# # num_list3=[0.821, 0.938, 0.860]   #SHEA
# # num_list4=[0.794, 0.911, 0.837]   #KAGNN
# #fr_en
# num_list=[0.914, 0.982, 0.940]    #RAGA-l
# num_list1=[0.924, 0.985, 0.947]   #TTEA-base
# num_list2=[0.952, 0.992, 0.968]   #TTEA-semi
# num_list3=[0.905, 0.970, 0.902]   #SHEA
# num_list4=[0.920, 0.976, 0.941]   #KAGNN
# x = np.arange(len(name_list))  # 标签位置
# width = 0.15  # 柱状图的宽度，可以根据自己的需求和审美来改
# # y_values=[0.0,0.2,0.4,0.6,0.8,1.0]   # y轴刻度
# fig, ax = plt.subplots()
# ###画标准差竖线 改这里可以调整柱状图的位置
# rects1 = ax.bar(x-2*width, num_list4, width,label='KAGNN')
# rects2 = ax.bar(x-width, num_list3, width,label='SHEA')
# rects3 = ax.bar(x, num_list, width,label='RAGA-base')
# rects4 = ax.bar(x+width , num_list1, width, label='TTEA-base')
# rects5 = ax.bar(x+2*width, num_list2, width,label='TTEA-semi')
# for a,b in zip(x - 2*width,num_list4):   #柱子上的数字显示
#     plt.text(a,b,'%.3f'%b,ha='center',va='bottom', fontsize=7)
# for a,b in zip(x-width,num_list3):   #柱子上的数字显示
#     plt.text(a,b,'%.3f'%b,ha='center',va='bottom', fontsize=7)
# for a,b in zip(x,num_list):   #柱子上的数字显示
#     plt.text(a,b,'%.3f'%b,ha='center',va='bottom', fontsize=7)
# for a, b in zip(x + width, num_list1):  # 柱子上的数字显示
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=7)
# for a,b in zip(x + 2*width,num_list2):   #柱子上的数字显示
#     plt.text(a,b,'%.3f'%b,ha='center',va='bottom', fontsize=7)
#
# # 为y轴、标题和x轴等添加一些文本。
# ax.set_ylabel('Performance', fontsize=12)   #y轴标题
# # ax.set_xlabel('X轴', fontsize=16)   #x轴标题
# ax.set_title('FR_EN',font1)  #标题
# ax.set_xticks(x)
# ax.set_xticklabels(name_list)
# # ax.legend(loc='upper center', ncol=5,prop=font2,framealpha=0.6,frameon=True)
# ax.legend(ncol=1,prop=font2,framealpha=0.3,frameon=True)
#
# plt.ylim(0,1.19)   #显示的刻度范围，比1稍微高一点
# fig.tight_layout()
# dir='D:\onedrive\OneDrive - tongji.edu.cn\桌面\期刊投稿\Alignment\基于三元组注意力和类型空间加强的跨语种实体对齐\论文'
# plt.savefig(dir+'\\FR_EN_bar.pdf', bbox_inches='tight')
# plt.show()
# exit(1)

#GCN层数比较
name_list = ['Hit@1', 'Hit@10', 'MRR']
# #zh_en
# num_list=[0.788, 0.916, 0.835]    #1gcn
# num_list1=[0.800, 0.934, 0.849]   #2gcn
# num_list2=[0.766, 0.935, 0.828]   #3gcn
#ja_en
num_list=[0.839, 0.939, 0.876]    #1gcn
num_list1=[0.837, 0.953, 0.880]   #2gcn
num_list2=[0.805, 0.948, 0.859]   #3gcn
# #fr_en
# num_list=[0.923, 0.979, 0.944]    #1gcn
# num_list1=[0.924, 0.985, 0.947]   #2gcn
# num_list2=[0.890, 0.977, 0.924]   #3gcn
x = np.arange(len(name_list))  # 标签位置
width = 0.20  # 柱状图的宽度，可以根据自己的需求和审美来改
# y_values=[0.0,0.2,0.4,0.6,0.8,1.0]   # y轴刻度
fig, ax = plt.subplots()
###画标准差竖线 改这里可以调整柱状图的位置
rects1 = ax.bar(x-width, num_list, width,label='l =1')
rects2 = ax.bar(x, num_list1, width,label='l =2')
rects3 = ax.bar(x+width, num_list2, width,label='l =3')
for a,b in zip(x-width,num_list):   #柱子上的数字显示
    plt.text(a,b,'%.3f'%b,ha='center',va='bottom', fontsize=8)
for a,b in zip(x,num_list1):   #柱子上的数字显示
    plt.text(a,b,'%.3f'%b,ha='center',va='bottom', fontsize=8)
for a, b in zip(x + width, num_list2):  # 柱子上的数字显示
    plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=8)

# 为y轴、标题和x轴等添加一些文本。
# ax.set_ylabel('Performance', fontsize=12)   #y轴标题
# ax.set_xlabel('X轴', fontsize=16)   #x轴标题
ax.set_title('JA_EN',font1)  #标题
ax.set_xticks(x)
ax.set_xticklabels(name_list)
# ax.legend(loc='upper center', ncol=5,prop=font2,framealpha=0.6,frameon=True)
ax.legend(ncol=1,prop=font2,framealpha=0.3,frameon=True)

plt.ylim(0,1.01)   #显示的刻度范围，比1稍微高一点
fig.tight_layout()
dir='D:\onedrive\OneDrive - tongji.edu.cn\桌面\期刊投稿\Alignment\基于三元组注意力和类型空间加强的跨语种实体对齐\论文'
plt.savefig(dir+'\\JA_EN_gcn_bar.pdf', bbox_inches='tight')
plt.show()
exit(1)

# #循环加强表示比较
# name_list = ['Hit@1', 'Hit@10', 'MRR']
# # # #zh_en
# # num_list=[0.798, 0.934,	0.847]    #无循环加强
# # num_list1=[0.799,	0.933,	0.848]    #互加强
# # num_list2=[0.800,	0.935,	0.849]    #循环互加强
# # #ja_en
# # num_list=[0.830,	0.950,	0.874]    #无循环加强
# # num_list1=[0.833,	0.948,	0.877]   #互加强
# # num_list2=[0.837,	0.953,	0.880]    #循环互加强
# #fr_en
# num_list=[0.922,	0.984,	0.945]    #无循环加强
# num_list1=[0.922,	0.984,	0.946]   #互加强
# num_list2=[0.924,	0.985,	0.947]    #循环互加强
# x = np.arange(len(name_list))  # 标签位置
# width = 0.20  # 柱状图的宽度，可以根据自己的需求和审美来改
# # y_values=[0.0,0.2,0.4,0.6,0.8,1.0]   # y轴刻度
# fig, ax = plt.subplots()
# ###画标准差竖线 改这里可以调整柱状图的位置
# rects1 = ax.bar(x-width, num_list, width,label='noncycle')
# rects2 = ax.bar(x, num_list1, width,label='co-enhanced')
# rects3 = ax.bar(x+width, num_list2, width,label='cycle co-enhanced')
# for a,b in zip(x-width,num_list):   #柱子上的数字显示
#     plt.text(a,b,'%.3f'%b,ha='center',va='bottom', fontsize=7)
# for a,b in zip(x,num_list1):   #柱子上的数字显示
#     plt.text(a,b,'%.3f'%b,ha='center',va='bottom', fontsize=7)
# for a, b in zip(x + width, num_list2):  # 柱子上的数字显示
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=7)
#
# # 为y轴、标题和x轴等添加一些文本。
# ax.set_ylabel('Performance', fontsize=12)   #y轴标题
# # ax.set_xlabel('X轴', fontsize=16)   #x轴标题
# ax.set_title('FR_EN',font1)  #标题
# ax.set_xticks(x)
# ax.set_xticklabels(name_list)
# # ax.legend(loc='upper center', ncol=5,prop=font2,framealpha=0.6,frameon=True)
# ax.legend(ncol=1,prop=font2,framealpha=0.3,frameon=True)
#
# plt.ylim(0,1.120)   #显示的刻度范围，比1稍微高一点
# fig.tight_layout()
# dir='D:\onedrive\OneDrive - tongji.edu.cn\桌面\期刊投稿\Alignment\基于三元组注意力和类型空间加强的跨语种实体对齐\论文'
# plt.savefig(dir+'\\FR_EN_xfusion_bar.pdf', bbox_inches='tight')
# plt.show()
# exit(1)