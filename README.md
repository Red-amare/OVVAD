开放词汇视频异常检测 (OVVAD) --V2

在main branch的基础上进行了更改

1、在train文件的main函数中调控消融实验和学习率：

 use_ta = True

 use_ski = True

 train_and_finetune_together = True（NAS）

 lr = 5e-4

2、可选model.py或者model_new.py作为模型。

其中model是最符合论文的模型，在ski模块直接拼接Fkonw和Xt。

model_new是效果较好的模型。在ski模块拼接后转为统一512维度输出。

3、用xd和ucf互为nas，同时也支持用xd_novel作xd本身的nas，ucf_novel作ucf本身的nas。方便进行实验。
