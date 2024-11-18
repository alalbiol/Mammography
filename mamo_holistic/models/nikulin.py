import torch
import torch.nn as nn
import torch.nn.functional as F


import tensorflow as tf
import numpy as np
import sys

class NikulinPatchModel(nn.Module):
    def __init__(self,**kwargs):
        super(NikulinPatchModel, self).__init__()

        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn5 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn7 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn9 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn11 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn13 = nn.BatchNorm2d(512)
        self.fc1 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0)
        self.bn14 = nn.BatchNorm2d(1024)
        self.fc2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.bn15 =  nn.BatchNorm2d(512)
        self.fc3 = nn.Conv2d(512, 5 , kernel_size=1, stride=1)
        # adaptive average pooling
        self.final_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self,x):
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        x = self.bn2(x)
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = self.bn3(x)
        x = F.relu(self.conv3(x))
        x = self.bn4(x)
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        x = self.bn5(x)
        x = F.relu(self.conv5(x))
        x = self.bn6(x)
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
                
        x = self.bn7(x)
        x = F.relu(self.conv7(x))
        x = self.bn8(x)
        x = F.relu(self.conv8(x))
        x = self.pool4(x)
        
        x = self.bn9(x)
        x = F.relu(self.conv9(x))
        x = self.bn10(x)
        x = F.relu(self.conv10(x))
        x = self.pool5(x)
        
        x = self.bn11(x)
        x = F.relu(self.conv11(x))
        x = self.bn12(x)
        x = F.relu(self.conv12(x))
        x = self.pool6(x)
                
        x = self.bn13(x)
        x = F.relu(self.fc1(x))
        x = self.bn14(x)
        x = F.relu(self.fc2(x))
        x = self.bn15(x)
        x = self.fc3(x)
        
        x = self.final_avg_pool(x)
        
        x = x.view(x.size(0), -1)
        return x




class NikulinImage(nn.Module):
    def __init__(self):
        super(NikulinImage, self).__init__()

        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn5 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn7 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn9 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn11 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn13 = nn.BatchNorm2d(512)
        self.fc1 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0)
        self.bn14 = nn.BatchNorm2d(1024)
        self.fc2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.bn15 =  nn.BatchNorm2d(512)
        self.fc3 = nn.Conv2d(512, 5 , kernel_size=1, stride=1)
        
        self.pool_final = nn.MaxPool2d(kernel_size=5, stride=5, padding=0)
        
        self.bn1_final = nn.BatchNorm2d(5)
        self.fc1_final = nn.Conv2d(5, 16, kernel_size=(3,2), stride=1, padding=0, bias=False)
        
        self.bn2_final = nn.BatchNorm2d(16)
        self.fc2_final = nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bn3_final = nn.BatchNorm2d(8)
        self.fc3_final = nn.Conv2d(8, 2, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bn_shortcut_final = nn.BatchNorm2d(5)
        self.shortcut_final = nn.Conv2d(5, 2, kernel_size=(3,2), stride=1, padding=0, bias=False)
        
    def load_weights_from_tf(self, checkpoint_path, exp_moving_avg=True, verbose=False):
        checkpoint = tf.train.load_checkpoint(checkpoint_path)

        for name, module in self.named_modules():
            if verbose:
                print(f"Module Name: {name}")
            if len(name) == 0:
                continue
            
            if "pool" in name:
                continue
            
            if name == "fc1_final":
                var_name = "TOWERS/fc1_final/weights"
                var_name = var_name + "/ExponentialMovingAverage" if exp_moving_avg else var_name
                var_tensor = checkpoint.get_tensor(var_name)
                var_tensor = torch.from_numpy(var_tensor).float().permute(3, 2, 0, 1)
                if var_tensor.shape == module.weight.shape:
                    module.weight.data = var_tensor
                else:
                    print("Weight shape does not match")
                    sys.exit()
            elif name == "bn1_final":
                tf_names = ["TOWERS/fc1_final/moving_variance", "TOWERS/fc1_final/moving_mean", \
                            "TOWERS/fc1_final/beta", "TOWERS/fc1_final/gamma"]
                
                torch_vars = ["running_var", "running_mean", "bias", "weight"]
                
                for tf_name, torch_var in zip(tf_names, torch_vars):
                    tf_name = tf_name + "/ExponentialMovingAverage" if exp_moving_avg else tf_name
                    var_tensor = checkpoint.get_tensor(tf_name)
                    var_tensor = torch.from_numpy(var_tensor).float()
                    if var_tensor.shape == getattr(module, torch_var).shape:
                        tmp = getattr(module, torch_var)
                        tmp.data = var_tensor
                    else:
                        print("Weight shape does not match")
                        sys.exit()
                                           
            elif name == "fc2_final":
                var_name = "TOWERS/fc2_final/weights"
                var_name = var_name + "/ExponentialMovingAverage" if exp_moving_avg else var_name
                var_tensor = checkpoint.get_tensor(var_name)
                var_tensor = torch.from_numpy(var_tensor).float().permute(3, 2, 0, 1)
                if var_tensor.shape == module.weight.shape:
                    module.weight.data = var_tensor
                else:
                    print("Weight shape does not match")
                    sys.exit()
            elif name == "bn2_final":
                tf_names = ["TOWERS/fc2_final/moving_variance", "TOWERS/fc2_final/moving_mean", \
                            "TOWERS/fc2_final/beta", "TOWERS/fc2_final/gamma"]
                
                torch_vars = ["running_var", "running_mean", "bias", "weight"]
                
                for tf_name, torch_var in zip(tf_names, torch_vars):
                    tf_name = tf_name + "/ExponentialMovingAverage" if exp_moving_avg else tf_name
                    var_tensor = checkpoint.get_tensor(tf_name)
                    var_tensor = torch.from_numpy(var_tensor).float()
                    if var_tensor.shape == getattr(module, torch_var).shape:
                        tmp = getattr(module, torch_var)
                        tmp.data = var_tensor
                    else:
                        print("Weight shape does not match")
                        sys.exit()
            elif name == "fc3_final":
                var_name = "TOWERS/fc3_final/weights"
                var_name = var_name + "/ExponentialMovingAverage" if exp_moving_avg else var_name
                var_tensor = checkpoint.get_tensor(var_name)
                var_tensor = torch.from_numpy(var_tensor).float().permute(3, 2, 0, 1)
                if var_tensor.shape == module.weight.shape:
                    module.weight.data = var_tensor
                else:
                    print("Weight shape does not match")
                    sys.exit()
            elif name == "bn3_final":
                tf_names = ["TOWERS/fc3_final/moving_variance", "TOWERS/fc3_final/moving_mean", \
                            "TOWERS/fc3_final/beta", "TOWERS/fc3_final/gamma"]
                
                torch_vars = ["running_var", "running_mean", "bias", "weight"]
                
                for tf_name, torch_var in zip(tf_names, torch_vars):
                    tf_name = tf_name + "/ExponentialMovingAverage" if exp_moving_avg else tf_name
                    var_tensor = checkpoint.get_tensor(tf_name)
                    var_tensor = torch.from_numpy(var_tensor).float()
                    if var_tensor.shape == getattr(module, torch_var).shape:
                        tmp = getattr(module, torch_var)
                        tmp.data = var_tensor
                    else:
                        print("Weight shape does not match")
                        sys.exit()            
            elif name == "shortcut_final":
                var_name = "TOWERS/shortcut_final/weights"
                var_name = var_name + "/ExponentialMovingAverage" if exp_moving_avg else var_name
                var_tensor = checkpoint.get_tensor(var_name)
                var_tensor = torch.from_numpy(var_tensor).float().permute(3, 2, 0, 1)
                if var_tensor.shape == module.weight.shape:
                    module.weight.data = var_tensor
                else:
                    print("Weight shape does not match")
                    sys.exit()
            elif name == "bn_shortcut_final":
                tf_names = ["TOWERS/shortcut_final/moving_variance", "TOWERS/shortcut_final/moving_mean", \
                            "TOWERS/shortcut_final/beta", "TOWERS/shortcut_final/gamma"]
                
                torch_vars = ["running_var", "running_mean", "bias", "weight"]
                
                for tf_name, torch_var in zip(tf_names, torch_vars):
                    tf_name = tf_name + "/ExponentialMovingAverage" if exp_moving_avg else tf_name
                    var_tensor = checkpoint.get_tensor(tf_name)
                    var_tensor = torch.from_numpy(var_tensor).float()
                    if var_tensor.shape == getattr(module, torch_var).shape:
                        tmp = getattr(module, torch_var)
                        tmp.data = var_tensor
                    else:
                        print("Weight shape does not match")
                        sys.exit()
                
            # Check if the module is an instance of Batch Normalization
            elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
                if verbose:
                    print(f"  Running Mean: {module.running_mean.shape}")
                    print(f"  Running Variance: {module.running_var.shape}")
                    print(f"  Gamma: {module.weight.shape}")
                    print(f"  Beta: {module.bias.shape}")
                
                gamma_name = f"TOWERS/{name}/gamma"
                gamma_name = gamma_name + "/ExponentialMovingAverage" if exp_moving_avg else gamma_name
                gamma = checkpoint.get_tensor(gamma_name)
                weight = torch.from_numpy(gamma).float()
                if weight.shape == module.weight.shape:
                    module.weight.data = weight
                else:
                    print("Gamma shape does not match")
                    sys.exit()
                
                beta_name = f"TOWERS/{name}/beta"
                beta_name = beta_name + "/ExponentialMovingAverage" if exp_moving_avg else beta_name
                beta = checkpoint.get_tensor(beta_name)
                bias = torch.from_numpy(beta).float()
                if beta.shape == module.bias.shape:
                    module.bias.data = bias
                else:
                    print("Beta shape does not match")
                    sys.exit()
                
                
                running_mean_name = f"TOWERS/{name}/moving_mean"
                running_mean_name = running_mean_name + "/ExponentialMovingAverage" if exp_moving_avg else running_mean_name
                running_mean = checkpoint.get_tensor(running_mean_name)
                running_mean = torch.from_numpy(running_mean).float()
                if running_mean.shape == module.running_mean.shape:
                    module.running_mean.data = running_mean
                else:
                    Exception("Running mean shape does not match")
                    
                running_var_name = f"TOWERS/{name}/moving_variance"
                running_var_name = running_var_name + "/ExponentialMovingAverage" if exp_moving_avg else running_var_name
                running_var = checkpoint.get_tensor(running_var_name)
                running_var = torch.from_numpy(running_var).float()
                if running_var.shape == module.running_var.shape:
                    module.running_var.data = running_var
                else:
                    Exception("Running variance shape does not match")
        
            elif isinstance(module, nn.Conv2d):
                if verbose:
                    print(f"  Weights: {module.weight.shape}")
                    print(f"  Biases: {module.bias.shape}")
                
                weight_name = f"TOWERS/{name}/weights"
                weight_name = weight_name + "/ExponentialMovingAverage" if exp_moving_avg else weight_name
                weight = checkpoint.get_tensor(weight_name)
                weight = torch.from_numpy(weight).float().permute(3, 2, 0, 1)
                # print(weight.shape)
                # print("weight shape ", weight.shape)
                # print("weisht module shape", module.weight.shape)
                # print(weight.shape == tuple(module.weight.shape))
                if weight.shape == tuple(module.weight.shape):
                    module.weight.data = weight
                else:
                    # generate exception
                    print("Weight shape does not match:", name)
                    print("weight shape ", weight.shape)
                    print("weight module shape", module.weight.shape)
                    sys.exit()
                    
                    
                bias_name = f"TOWERS/{name}/biases"
                bias_name = bias_name + "/ExponentialMovingAverage" if exp_moving_avg else bias_name
                bias = checkpoint.get_tensor(bias_name)
                bias = torch.from_numpy(bias).float()
                
                if bias.shape == module.bias.shape:
                    module.bias.data = bias
                else:
                    print("Bias shape does not match:", name)
                    sys.exit()
                
            else:
                print("Unknonw module:", name)
                sys.exit()
                    
                
                
            # # Check if the module has parameters (tensors)
            # if len(list(module.parameters())) > 0:
            #     for param_name, param in module.named_parameters():
            #         print(f"  Parameter Name: {param_name}")
            #         print(f"  Parameter shape: {param.shape}")
            # else:
            #     print("  No parameters in this module")

            # print("=" * 50)        



    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        x = self.bn2(x)
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = self.bn3(x)
        x = F.relu(self.conv3(x))
        x = self.bn4(x)
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        x = self.bn5(x)
        x = F.relu(self.conv5(x))
        x = self.bn6(x)
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
                
        x = self.bn7(x)
        x = F.relu(self.conv7(x))
        x = self.bn8(x)
        x = F.relu(self.conv8(x))
        x = self.pool4(x)
        
        x = self.bn9(x)
        x = F.relu(self.conv9(x))
        x = self.bn10(x)
        x = F.relu(self.conv10(x))
        x = self.pool5(x)
        
        x = self.bn11(x)
        x = F.relu(self.conv11(x))
        x = self.bn12(x)
        x = F.relu(self.conv12(x))
        x = self.pool6(x)
                
        x = self.bn13(x)
        x = F.relu(self.fc1(x))
        x = self.bn14(x)
        x = F.relu(self.fc2(x))
        x = self.bn15(x)
        x = self.fc3(x)
        
        x_pool = self.pool_final(x)
        
        x = F.relu(self.bn1_final(x_pool))
        x = self.fc1_final(x)
        
        x = F.relu(self.bn2_final(x))
        x = self.fc2_final(x)
        
        x = F.relu(self.bn3_final(x))
        x = self.fc3_final(x)
        
        shorcut = self.bn_shortcut_final(x_pool)
        shorcut = self.shortcut_final(shorcut)
        
        final_logits = x + shorcut
        
        return final_logits[:, :, 0, 0]



if __name__ == "__main__":
    # Instantiate the model
    model = MammoModel()  # Make sure to replace YourOptionsClass with the actual class you're using for options

    # Print the model architecture
    #for param in model.named_parameters():
    #    print(param[0], param[1].shape)


    model.load_weights_from_tf('./data/SC1_nets/Final_Round_RC2/best_model.ckpt-5000')


    # for name, module in model.named_modules():
    #     print(f"Module Name: {name}")

    #     # Check if the module is an instance of Batch Normalization
    #     if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
    #         print(f"  Running Mean: {module.running_mean.shape}")
    #         print(f"  Running Variance: {module.running_var.shape}")

    #     # Check if the module has parameters (tensors)
    #     if len(list(module.parameters())) > 0:
    #         for param_name, param in module.named_parameters():
    #             print(f"  Parameter Name: {param_name}")
    #             print(f"  Parameter shape: {param.shape}")
    #     else:
    #         print("  No parameters in this module")

    #     print("=" * 50)
