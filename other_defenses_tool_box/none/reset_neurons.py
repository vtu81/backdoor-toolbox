
def reset(model,arch,selected_neuros,freeze = False):
    model.eval()
    channel_indices=[]
    
    if arch == "vgg16":
    
        conv_list = ["Conv_0_","Conv_2_","Conv_5_","Conv_7_","Conv_10_","Conv_12_","Conv_14_","Conv_17_","Conv_19_","Conv_21_","Conv_24_","Conv_26_","Conv_28_"]
        layer_list = [0,2,5,7,10,12,14,17,19,21,24,26,28]
        
        for j in range(len(layer_list)):
            # print("reset Conv "+str(layer_list[j])+":")
            channel_indices=[]
            for line in selected_neuros:
                layer = line.split("_")[1]
                neuro = line.split("_")[2]
                if layer == str(layer_list[j]):
                    channel_indices.append(int(neuro))
                    # print(int(neuro))
            for name, param in model.named_parameters():
                for i in range(0,len(channel_indices)):
                    channel = channel_indices[i]
                    if name == "features."+str(layer_list[j])+".weight":
                        param.data[channel,:,:,:] = 0
                    if name == "features."+str(layer_list[j])+".bias":
                        param.data[channel] = 0
                    
                    if (j == (len(layer_list)-1)):
                        if name == "classifier.weight":
                            param.data[:,channel] = 0
                    else:
                        if name == "features."+str(layer_list[j+1])+".weight":
                            param.data[:,channel,:,:] = 0
    
    elif arch == "nin":
    
        # print("reset layer 0:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[1]
            neuro = line.split("_")[2]
            if layer == "0":
                channel_indices.append(int(neuro))
                # print(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "classifier.0.weight":
                    param.data[channel,:,:,:] = 0
                if name == "classifier.0.bias":
                    param.data[channel] = 0
                if name == "classifier.2.weight":
                    param.data[:,channel,:,:] = 0
                    
            if freeze:
                for i in range(0,len(channel_indices)):
                    channel = channel_indices[i]
                    if name == "classifier.0.weight":
                        param.data[channel,:,:,:] = 0
                        param.requires_grad = False
                    if name == "classifier.0.bias":
                        param.data[channel] = 0
                        param.requires_grad = False
                    if name == "classifier.2.weight":
                        param.data[:,channel,:,:] = 0
                        param.requires_grad = False
    
        # print("reset layer 2:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[1]
            neuro = line.split("_")[2]
            if layer == "2":
                channel_indices.append(int(neuro))
                # print(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "classifier.2.weight":
                    param.data[channel,:,:,:] = 0
                if name == "classifier.2.bias":
                    param.data[channel] = 0
                if name == "classifier.4.weight":
                    param.data[:,channel,:,:] = 0
            
            if freeze:
                for i in range(0,len(channel_indices)):
                    channel = channel_indices[i]
                    if name == "classifier.0.weight":
                        param.data[channel,:,:,:] = 0
                        param.requires_grad = False
                    if name == "classifier.0.bias":
                        param.data[channel] = 0
                        param.requires_grad = False
                    if name == "classifier.2.weight":
                        param.data[:,channel,:,:] = 0
                        param.requires_grad = False
    
        # print("reset layer 4:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[1]
            neuro = line.split("_")[2]
            if layer == "4":
                channel_indices.append(int(neuro))
                # print(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "classifier.4.weight":
                    param.data[channel,:,:,:] = 0
                if name == "classifier.4.bias":
                    param.data[channel] = 0
                if name == "classifier.8.weight":
                    param.data[:,channel,:,:] = 0
                    
            if freeze:
                for i in range(0,len(channel_indices)):
                    channel = channel_indices[i]
                    if name == "classifier.4.weight":
                        param.data[channel,:,:,:] = 0
                        param.requires_grad = False
                    if name == "classifier.4.bias":
                        param.data[channel] = 0
                        param.requires_grad = False
                    if name == "classifier.8.weight":
                        param.data[:,channel,:,:] = 0
                        param.requires_grad = False
    
        # print("reset layer 8:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[1]
            neuro = line.split("_")[2]
            if layer == "8":
                channel_indices.append(int(neuro))
                # print(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "classifier.8.weight":
                    param.data[channel,:,:,:] = 0
                if name == "classifier.8.bias":
                    param.data[channel] = 0
                if name == "classifier.10.weight":
                    param.data[:,channel,:,:] = 0
                    
            if freeze:
                for i in range(0,len(channel_indices)):
                    channel = channel_indices[i]
                    if name == "classifier.8.weight":
                        param.data[channel,:,:,:] = 0
                        param.requires_grad = False
                    if name == "classifier.8.bias":
                        param.data[channel] = 0
                        param.requires_grad = False
                    if name == "classifier.10.weight":
                        param.data[:,channel,:,:] = 0
                        param.requires_grad = False

        # print("reset layer 10:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[1]
            neuro = line.split("_")[2]
            if layer == "10":
                channel_indices.append(int(neuro))
                # print(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "classifier.10.weight":
                    param.data[channel,:,:,:] = 0
                if name == "classifier.10.bias":
                    param.data[channel] = 0
                if name == "classifier.12.weight":
                    param.data[:,channel,:,:] = 0
            
            if freeze:
                for i in range(0,len(channel_indices)):
                    channel = channel_indices[i]
                    if name == "classifier.10.weight":
                        param.data[channel,:,:,:] = 0
                        param.requires_grad = False
                    if name == "classifier.10.bias":
                        param.data[channel] = 0
                        param.requires_grad = False
                    if name == "classifier.12.weight":
                        param.data[:,channel,:,:] = 0
                        param.requires_grad = False

        # print("reset layer 12:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[1]
            neuro = line.split("_")[2]
            if layer == "12":
                channel_indices.append(int(neuro))
                # print(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "classifier.12.weight":
                    param.data[channel,:,:,:] = 0
                if name == "classifier.12.bias":
                    param.data[channel] = 0
                if name == "classifier.16.weight":
                    param.data[:,channel,:,:] = 0
                    
            if freeze:
                for i in range(0,len(channel_indices)):
                    channel = channel_indices[i]
                    if name == "classifier.12.weight":
                        param.data[channel,:,:,:] = 0
                        param.requires_grad = False
                    if name == "classifier.12.bias":
                        param.data[channel] = 0
                        param.requires_grad = False
                    if name == "classifier.16.weight":
                        param.data[:,channel,:,:] = 0
                        param.requires_grad = False
                    
        # print("reset layer 16:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[1]
            neuro = line.split("_")[2]
            if layer == "16":
                channel_indices.append(int(neuro))
                # print(int(neuro))
        for name, param in model.named_parameters():

            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "classifier.16.weight":
                    param.data[channel,:,:,:] = 0
                if name == "classifier.16.bias":
                    param.data[channel] = 0
                if name == "classifier.18.weight":
                    param.data[:,channel,:,:] = 0
            
            if freeze:
                for i in range(0,len(channel_indices)):
                    channel = channel_indices[i]
                    if name == "classifier.16.weight":
                        param.data[channel,:,:,:] = 0
                        param.requires_grad = False
                    if name == "classifier.16.bias":
                        param.data[channel] = 0
                        param.requires_grad = False
                    if name == "classifier.18.weight":
                        param.data[:,channel,:,:] = 0
                        param.requires_grad = False
        
        # print("reset layer 18:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[1]
            neuro = line.split("_")[2]
            if layer == "18":
                channel_indices.append(int(neuro))
                # print(int(neuro))

        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "classifier.18.weight":
                    param.data[channel,:,:,:] = 0
                if name == "classifier.18.bias":
                    param.data[channel] = 0
                if name == "classifier.20.weight":
                    param.data[:,channel,:,:] = 0
                    
            if freeze:
                for i in range(0,len(channel_indices)):
                    channel = channel_indices[i]
                    if name == "classifier.18.weight":
                        param.data[channel,:,:,:] = 0
                        param.requires_grad = False
                    if name == "classifier.18.bias":
                        param.data[channel] = 0
                        param.requires_grad = False
                    if name == "classifier.20.weight":
                        param.data[:,channel,:,:] = 0
                        param.requires_grad = False
    
    elif arch in ["resnet18","resnet18_leakyrelu","resnet18_elu","resnet18_prelu","resnet18_tanhshrink","resnet18_softplus"]:
    
        # print("reset Conv1:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[0]
            sublayer = line.split("_")[1]
            conv = line.split("_")[2]
            neuro = line.split("_")[3]
            if (layer == "Conv1") and (sublayer == "Conv1") and (conv == "Conv1"):
                channel_indices.append(int(neuro))
                # print(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "conv1.weight":
                    param.data[channel,:,:,:] = 0
                if name == "bn1.weight":
                    param.data[channel] = 0
                if name == "bn1.bias":
                    param.data[channel] = 0
                if name == "layer1.0.conv1.weight":
                    param.data[:,channel,:,:] = 0

        # print("reset Layer1.0.Conv1:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[0]
            sublayer = line.split("_")[1]
            conv = line.split("_")[2]
            neuro = line.split("_")[3]
            
            if (layer == "Layer1") and (sublayer == "0") and (conv == "Conv1"):
                channel_indices.append(int(neuro))
                # print(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "layer1.0.conv1.weight":
                    param.data[channel,:,:,:] = 0
                if name == "layer1.0.bn1.weight":
                    param.data[channel] = 0
                if name == "layer1.0.bn1.bias":
                    param.data[channel] = 0
                if name == "layer1.0.conv2.weight":
                    param.data[:,channel,:,:] = 0

        # print("reset Layer1.0.Conv2:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[0]
            sublayer = line.split("_")[1]
            conv = line.split("_")[2]
            neuro = line.split("_")[3]
            
            if (layer == "Layer1") and (sublayer == "0") and (conv == "Conv2"):
                channel_indices.append(int(neuro))
                # print(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "layer1.0.conv2.weight":
                    param.data[channel,:,:,:] = 0
                if name == "layer1.0.bn2.weight":
                    param.data[channel] = 0
                if name == "layer1.0.bn2.bias":
                    param.data[channel] = 0
                if name == "layer1.1.conv1.weight":
                    param.data[:,channel,:,:] = 0
    
        # print("reset Layer1.1.Conv1:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[0]
            sublayer = line.split("_")[1]
            conv = line.split("_")[2]
            neuro = line.split("_")[3]

            if (layer == "Layer1") and (sublayer == "1") and (conv == "Conv1"):
                channel_indices.append(int(neuro))
                # print(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "layer1.1.conv1.weight":
                    param.data[channel,:,:,:] = 0
                if name == "layer1.1.bn1.weight":
                    param.data[channel] = 0
                if name == "layer1.1.bn1.bias":
                    param.data[channel] = 0
                
                if name == "layer1.1.conv2.weight":
                    param.data[:,channel,:,:] = 0
    
        # print("reset Layer1.1.Conv2:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[0]
            sublayer = line.split("_")[1]
            conv = line.split("_")[2]
            neuro = line.split("_")[3]
            
            if (layer == "Layer1") and (sublayer == "1") and (conv == "Conv2"):
                channel_indices.append(int(neuro))
                # print(int(neuro))

        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "layer1.1.conv2.weight":
                    param.data[channel,:,:,:] = 0
                if name == "layer1.1.bn2.weight":
                    param.data[channel] = 0
                if name == "layer1.1.bn2.bias":
                    param.data[channel] = 0
                ##########################################
                if name == "layer2.0.conv1.weight":
                    param.data[:,channel,:,:] = 0
    
        # print("reset Layer2.0.Conv1:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[0]
            sublayer = line.split("_")[1]
            conv = line.split("_")[2]
            neuro = line.split("_")[3]
            
            if (layer == "Layer2") and (sublayer == "0") and (conv == "Conv1"):
                channel_indices.append(int(neuro))
                # print(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "layer2.0.conv1.weight":
                    param.data[channel,:,:,:] = 0
                if name == "layer2.0.bn1.weight":
                    param.data[channel] = 0
                if name == "layer2.0.bn1.bias":
                    param.data[channel] = 0
                if name == "layer2.0.conv2.weight":
                    param.data[:,channel,:,:] = 0

                    
        # print("reset Layer2.0.Downsample:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[0]
            sublayer = line.split("_")[1]
            conv = line.split("_")[2]
            neuro = line.split("_")[3]
            
            if (layer == "Layer2") and (sublayer == "0") and (conv == "Downsample"):
                channel_indices.append(int(neuro))
                # print(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "layer2.0.downsample.0.weight":
                    param.data[channel,:,:,:] = 0
                if name == "layer2.0.downsample.1.weight":
                    param.data[channel] = 0
                if name == "layer2.0.downsample.1.bias":
                    param.data[channel] = 0

        # print("reset Layer2.0.Conv2:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[0]
            sublayer = line.split("_")[1]
            conv = line.split("_")[2]
            neuro = line.split("_")[3]
            
            if (layer == "Layer2") and (sublayer == "0") and (conv == "Conv2"):
                channel_indices.append(int(neuro))
                # print(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "layer2.0.conv2.weight":
                    param.data[channel,:,:,:] = 0
                if name == "layer2.0.bn2.weight":
                    param.data[channel] = 0
                if name == "layer2.0.bn2.bias":
                    param.data[channel] = 0
                if name == "layer2.1.conv1.weight":
                    param.data[:,channel,:,:] = 0
    
        # print("reset Layer2.1.Conv1:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[0]
            sublayer = line.split("_")[1]
            conv = line.split("_")[2]
            neuro = line.split("_")[3]

            if (layer == "Layer2") and (sublayer == "1") and (conv == "Conv1"):
                channel_indices.append(int(neuro))
                # print(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "layer2.1.conv1.weight":
                    param.data[channel,:,:,:] = 0
                if name == "layer2.1.bn1.weight":
                    param.data[channel] = 0
                if name == "layer2.1.bn1.bias":
                    param.data[channel] = 0
                
                if name == "layer2.1.conv2.weight":
                    param.data[:,channel,:,:] = 0
    
        # print("reset Layer2.1.Conv2:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[0]
            sublayer = line.split("_")[1]
            conv = line.split("_")[2]
            neuro = line.split("_")[3]
            
            if (layer == "Layer2") and (sublayer == "1") and (conv == "Conv2"):
                channel_indices.append(int(neuro))
                # print(int(neuro))

        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "layer2.1.conv2.weight":
                    param.data[channel,:,:,:] = 0
                if name == "layer2.1.bn2.weight":
                    param.data[channel] = 0
                if name == "layer2.1.bn2.bias":
                    param.data[channel] = 0
                ##########################################
                if name == "layer3.0.conv1.weight":
                    param.data[:,channel,:,:] = 0
    
        # print("reset Layer3.0.Conv1:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[0]
            sublayer = line.split("_")[1]
            conv = line.split("_")[2]
            neuro = line.split("_")[3]
            
            if (layer == "Layer3") and (sublayer == "0") and (conv == "Conv1"):
                channel_indices.append(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "layer3.0.conv1.weight":
                    param.data[channel,:,:,:] = 0
                if name == "layer3.0.bn1.weight":
                    param.data[channel] = 0
                if name == "layer3.0.bn1.bias":
                    param.data[channel] = 0

                if name == "layer3.0.conv2.weight":
                    param.data[:,channel,:,:] = 0
                    
        # print("reset Layer3.0.Downsample:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[0]
            sublayer = line.split("_")[1]
            conv = line.split("_")[2]
            neuro = line.split("_")[3]
            
            if (layer == "Layer3") and (sublayer == "0") and (conv == "Downsample"):
                channel_indices.append(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "layer3.0.downsample.0.weight":
                    param.data[channel,:,:,:] = 0
                if name == "layer3.0.downsample.1.weight":
                    param.data[channel] = 0
                if name == "layer3.0.downsample.1.bias":
                    param.data[channel] = 0

        # print("reset Layer3.0.Conv2:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[0]
            sublayer = line.split("_")[1]
            conv = line.split("_")[2]
            neuro = line.split("_")[3]
            
            if (layer == "Layer3") and (sublayer == "0") and (conv == "Conv2"):
                channel_indices.append(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "layer3.0.conv2.weight":
                    param.data[channel,:,:,:] = 0
                if name == "layer3.0.bn2.weight":
                    param.data[channel] = 0
                if name == "layer3.0.bn2.bias":
                    param.data[channel] = 0

                if name == "layer3.1.conv1.weight":
                    param.data[:,channel,:,:] = 0

        # print("reset Layer3.1.Conv1:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[0]
            sublayer = line.split("_")[1]
            conv = line.split("_")[2]
            neuro = line.split("_")[3]
            
            
            if (layer == "Layer3") and (sublayer == "1") and (conv == "Conv1"):
                channel_indices.append(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "layer3.1.conv1.weight":
                    param.data[channel,:,:,:] = 0
                if name == "layer3.1.bn1.weight":
                    param.data[channel] = 0
                if name == "layer3.1.bn1.bias":
                    param.data[channel] = 0
     
                if name == "layer3.1.conv2.weight":
                    param.data[:,channel,:,:] = 0

        # print("reset Layer3.1.Conv2:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[0]
            sublayer = line.split("_")[1]
            conv = line.split("_")[2]
            neuro = line.split("_")[3]
            
            if (layer == "Layer3") and (sublayer == "1") and (conv == "Conv2"):
                channel_indices.append(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "layer3.1.conv2.weight":
                    param.data[channel,:,:,:] = 0
                if name == "layer3.1.bn2.weight":
                    param.data[channel] = 0
                if name == "layer3.1.bn2.bias":
                    param.data[channel] = 0
                if name == "layer4.0.conv1.weight":
                    param.data[:,channel,:,:] = 0

        # print("reset Layer4.0.Conv1:")
        for line in selected_neuros:
            layer = line.split("_")[0]
            sublayer = line.split("_")[1]
            conv = line.split("_")[2]
            neuro = line.split("_")[3]
            
            if (layer == "Layer4") and (sublayer == "0") and (conv == "Conv1"):
                channel_indices.append(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "layer4.0.conv1.weight":
                    param.data[channel,:,:,:] = 0
                if name == "layer4.0.bn1.weight":
                    param.data[channel] = 0
                if name == "layer4.0.bn1.bias":
                    param.data[channel] = 0
                    
                if name == "layer4.0.conv2.weight":
                    param.data[:,channel,:,:] = 0   


        # print("reset Layer4.0.Downsample:")
        channel_indices=[]
        for line in selected_neuros:
            layer = line.split("_")[0]
            sublayer = line.split("_")[1]
            conv = line.split("_")[2]
            neuro = line.split("_")[3]
            
            if (layer == "Layer4") and (sublayer == "0") and (conv == "Downsample"):
                channel_indices.append(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "layer4.0.downsample.0.weight":
                    param.data[channel,:,:,:] = 0
                
                if name == "layer4.0.downsample.1.weight":
                    param.data[channel] = 0
                if name == "layer4.0.downsample.1.bias":
                    param.data[channel] = 0

        # print("reset Layer4.0.Conv2:")
        for line in selected_neuros:
            layer = line.split("_")[0]
            sublayer = line.split("_")[1]
            conv = line.split("_")[2]
            neuro = line.split("_")[3]
            
            if (layer == "Layer4") and (sublayer == "0") and (conv == "Conv2"):
                channel_indices.append(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "layer4.0.conv2.weight":
                    param.data[channel,:,:,:] = 0
                if name == "layer4.0.bn2.weight":
                    param.data[channel] = 0
                if name == "layer4.0.bn2.bias":
                    param.data[channel] = 0
                    
                if name == "layer4.1.conv1.weight":
                    param.data[:,channel,:,:] = 0   

        # print("reset Layer4.1.Conv1:")
        for line in selected_neuros:
            layer = line.split("_")[0]
            sublayer = line.split("_")[1]
            conv = line.split("_")[2]
            neuro = line.split("_")[3]
            
            if (layer == "Layer4") and (sublayer == "1") and (conv == "Conv1"):
                channel_indices.append(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "layer4.1.conv1.weight":
                    param.data[channel,:,:,:] = 0
                if name == "layer4.1.bn1.weight":
                    param.data[channel] = 0
                if name == "layer4.1.bn1.bias":
                    param.data[channel] = 0
                    
                if name == "layer4.1.conv2.weight":
                    param.data[:,channel,:,:] = 0   

        # print("reset Layer4.1.Conv2:")
        for line in selected_neuros:
            layer = line.split("_")[0]
            sublayer = line.split("_")[1]
            conv = line.split("_")[2]
            neuro = line.split("_")[3]
            
            if (layer == "Layer4") and (sublayer == "1") and (conv == "Conv2"):
                channel_indices.append(int(neuro))
        for name, param in model.named_parameters():
            for i in range(0,len(channel_indices)):
                channel = channel_indices[i]
                if name == "layer4.1.conv2.weight":
                    param.data[channel,:,:,:] = 0
                if name == "layer4.1.bn2.weight":
                    param.data[channel] = 0
                if name == "layer4.1.bn2.bias":
                    param.data[channel] = 0
                    
                if name == "fc.weight":
                    param.data[:,channel] = 0   
                    
    return model
