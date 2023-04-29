import numpy as np
import math
from torch.autograd import Variable
import torch.nn.functional as F

def analyze_neuros(model,arch,max_reset_fraction,lamda_l,lamda_h, \
                   num_classes,num_for_detect_biased,trainloader_no_shuffle):
    model.eval()
    selected_neuros = []
    index = 0
    #loader = trainloader
    loader = trainloader_no_shuffle
    #max_reset_fraction = args.reset_max_per_layer_fraction
    
    if arch == "nin":
        conv_list = ["Conv_0_","Conv_2_","Conv_4_","Conv_8_","Conv_10_","Conv_12_","Conv_16_","Conv_18_"]
        channel_num_list = [192,160,96,192,192,192,192,192]
    elif arch == "vgg16":
        conv_list = ["Conv_0_","Conv_2_","Conv_5_","Conv_7_","Conv_10_","Conv_12_","Conv_14_","Conv_17_","Conv_19_","Conv_21_","Conv_24_","Conv_26_","Conv_28_"]
        channel_num_list = [64,64,128,128,256,256,256,512,512,512,512,512,512]
    elif arch in ["resnet18","resnet18_leakyrelu","resnet18_elu","resnet18_prelu","resnet18_tanhshrink","resnet18_softplus"]:
        conv_list = ["Layer2_0_Conv1_","Layer2_0_Downsample_","Layer2_0_Conv2_","Layer2_1_Conv1_","Layer2_1_Conv2_","Layer3_0_Conv1_","Layer3_0_Downsample_","Layer3_0_Conv2_","Layer3_1_Conv1_","Layer3_1_Conv2_","Layer4_0_Conv1_","Layer4_0_Downsample_","Layer4_0_Conv2_","Layer4_1_Conv1_","Layer4_1_Conv2_"]
        channel_num_list = [128,128,128,128,128,256,256,256,256,256,512,512,512,512,512]

    imgs_high_activation_times = np.zeros([len(loader.dataset)])
    imgs_high_activation_times_counter = np.zeros([len(loader.dataset)])
    conv_activation_all_list = []

    for i in range(len(conv_list)):
        channel_num = channel_num_list[i]
        conv_activation_all_list.append(np.zeros([len(loader.dataset),channel_num]))
    
    output_all = np.zeros([len(loader.dataset),num_classes])
    
    counter = 0
    
    if arch in ["resnet18","resnet18_leakyrelu","resnet18_elu","resnet18_prelu","resnet18_tanhshrink","resnet18_softplus","efficientnet","mobilenetv2","densenet121"]:
        for data, target in loader:
            data, target = Variable(data.cuda()), Variable(target.cuda())
            output = model(data)
            pred = F.softmax(output, dim=1)
            batch_size = data.shape[0]
            output_all[counter:counter+batch_size] = pred.cpu().detach().numpy()
            for i in range(len(conv_list)):
                conv_name = conv_list[i]
                if "FC" in conv_name:
                    conv_activation_all_list[i][counter:counter+batch_size] = model.inter_feature[conv_name].cpu().detach().numpy()
                else:
                    conv_activation_all_list[i][counter:counter+batch_size] = model.inter_feature[conv_name].max(-1).values.max(-1).values.cpu().detach().numpy()
            counter = counter+batch_size
            # print(counter)
    
    else:
        target_count = [0]*10
        for data, target in loader:
            data, target = Variable(data.cuda()), Variable(target.cuda())
            output, inner_output_list = model.get_all_inner_activation(data)
            pred = F.softmax(output, dim=1)
            for i in range(target.shape[0]):
                target_count[target[i].item()] = target_count[target[i].item()] + 1
            batch_size = data.shape[0]
            output_all[counter:counter+batch_size] = pred.cpu().detach().numpy()
            for i in range(len(conv_list)):
                conv_name = conv_list[i]
                if "FC" in conv_name:
                    conv_activation_all_list[i][counter:counter+batch_size] = inner_output_list[i].cpu().detach().numpy()
                else:
                    conv_activation_all_list[i][counter:counter+batch_size] = inner_output_list[i].max(-1).values.max(-1).values.cpu().detach().numpy()
            counter = counter+batch_size
        #     print(counter)
        # print(target_count)

    diff_class_channel_numpy_list = []
    for i in range(len(conv_list)):
        channel_num = channel_num_list[i]
        diff_class_channel_numpy_list.append(np.zeros([channel_num,num_classes]))

    for i in range(len(conv_list)):
        conv_name = conv_list[i]
        channel_num = channel_num_list[i]
        # print(conv_name)
        # print(channel_num)
        for j in range(channel_num):
            for k in range(num_classes):
                strong_output_indexs = np.where(output_all[:,k]>lamda_h)
                weak_output_indexs = np.where(output_all[:,k]<lamda_l)
                if strong_output_indexs[0].shape[0] == 0:
                    continue

                strong_output_conv_activation_max = conv_activation_all_list[i][strong_output_indexs,j].max()
                statistics_for_base_distribution = np.mean(conv_activation_all_list[i][weak_output_indexs,j]) + 3*np.std(conv_activation_all_list[i][weak_output_indexs,j])
                diff = strong_output_conv_activation_max - statistics_for_base_distribution
                diff_class_channel_numpy_list[i][j,k] = diff
                
        diff_channel_numpy = diff_class_channel_numpy_list[i].max(1)
        channel_sorted = diff_channel_numpy.argsort()[::-1]

        max_num = math.ceil(max_reset_fraction*channel_num)
        top_channel = channel_sorted[:max_num]

        if num_for_detect_biased == -1:
            top_channel_calculate_biased_imgs = channel_sorted[:1]
        else:
            top_channel_calculate_biased_imgs = channel_sorted[:math.ceil(num_for_detect_biased*channel_num)]

        for m in top_channel:
            selected_neuros.append(conv_name+str(m))
            
        for j in top_channel_calculate_biased_imgs:
            # print(j)
            for k in range(num_classes):
                strong_output_indexs = np.where(output_all[:,k]>lamda_h)
                weak_output_indexs = np.where(output_all[:,k]<lamda_l)
                imgs_high_activation_times[:] = imgs_high_activation_times[:] + conv_activation_all_list[i][:,j]
    
    print("Require jenkspy==0.2.0")
    import jenkspy
    breaks = jenkspy.jenks_breaks(imgs_high_activation_times.reshape(-1, 1), nb_class=2)
    # print(breaks)
    poison_sample_index = np.where(imgs_high_activation_times>=breaks[1])[0].tolist()
    print("number of detected Trojan samples:",len(poison_sample_index))
    return selected_neuros, poison_sample_index
