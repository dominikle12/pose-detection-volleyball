import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

# Use JIT-friendly layer creation
def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))

class bodypose_model(nn.Module):
    def __init__(self, use_half_precision=False):
        super(bodypose_model, self).__init__()
        
        # Track if we're using half precision
        self.use_half_precision = use_half_precision
        
        # Cache the no_relu_layers list
        self.no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1',
                          'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2',
                          'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1',
                          'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
        
        # Base network definition
        block0 = OrderedDict([
                  ('conv1_1', [3, 64, 3, 1, 1]),
                  ('conv1_2', [64, 64, 3, 1, 1]),
                  ('pool1_stage1', [2, 2, 0]),
                  ('conv2_1', [64, 128, 3, 1, 1]),
                  ('conv2_2', [128, 128, 3, 1, 1]),
                  ('pool2_stage1', [2, 2, 0]),
                  ('conv3_1', [128, 256, 3, 1, 1]),
                  ('conv3_2', [256, 256, 3, 1, 1]),
                  ('conv3_3', [256, 256, 3, 1, 1]),
                  ('conv3_4', [256, 256, 3, 1, 1]),
                  ('pool3_stage1', [2, 2, 0]),
                  ('conv4_1', [256, 512, 3, 1, 1]),
                  ('conv4_2', [512, 512, 3, 1, 1]),
                  ('conv4_3_CPM', [512, 256, 3, 1, 1]),
                  ('conv4_4_CPM', [256, 128, 3, 1, 1])
              ])

        # Stage 1
        block1_1 = OrderedDict([
                    ('conv5_1_CPM_L1', [128, 128, 3, 1, 1]),
                    ('conv5_2_CPM_L1', [128, 128, 3, 1, 1]),
                    ('conv5_3_CPM_L1', [128, 128, 3, 1, 1]),
                    ('conv5_4_CPM_L1', [128, 512, 1, 1, 0]),
                    ('conv5_5_CPM_L1', [512, 38, 1, 1, 0])
                ])

        block1_2 = OrderedDict([
                    ('conv5_1_CPM_L2', [128, 128, 3, 1, 1]),
                    ('conv5_2_CPM_L2', [128, 128, 3, 1, 1]),
                    ('conv5_3_CPM_L2', [128, 128, 3, 1, 1]),
                    ('conv5_4_CPM_L2', [128, 512, 1, 1, 0]),
                    ('conv5_5_CPM_L2', [512, 19, 1, 1, 0])
                ])
                
        # Create model0 first to enable fusion of layers
        self.model0 = make_layers(block0, self.no_relu_layers)
        self.model1_1 = make_layers(block1_1, self.no_relu_layers)
        self.model1_2 = make_layers(block1_2, self.no_relu_layers)
        
        # Create subsequent stage models
        blocks = {}
        for i in range(2, 7):
            blocks['model%d_1' % i] = make_layers(OrderedDict([
                ('Mconv1_stage%d_L1' % i, [185, 128, 7, 1, 3]),
                ('Mconv2_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv3_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv4_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv5_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv6_stage%d_L1' % i, [128, 128, 1, 1, 0]),
                ('Mconv7_stage%d_L1' % i, [128, 38, 1, 1, 0])
            ]), self.no_relu_layers)
            
            blocks['model%d_2' % i] = make_layers(OrderedDict([
                ('Mconv1_stage%d_L2' % i, [185, 128, 7, 1, 3]),
                ('Mconv2_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv3_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv4_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv5_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv6_stage%d_L2' % i, [128, 128, 1, 1, 0]),
                ('Mconv7_stage%d_L2' % i, [128, 19, 1, 1, 0])
            ]), self.no_relu_layers)
        
        # Directly assign models from blocks
        self.model2_1 = blocks['model2_1']
        self.model3_1 = blocks['model3_1']
        self.model4_1 = blocks['model4_1']
        self.model5_1 = blocks['model5_1']
        self.model6_1 = blocks['model6_1']

        self.model2_2 = blocks['model2_2']
        self.model3_2 = blocks['model3_2']
        self.model4_2 = blocks['model4_2']
        self.model5_2 = blocks['model5_2']
        self.model6_2 = blocks['model6_2']
        
        # Initialize weights for faster convergence
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Apply half precision if enabled
        if self.use_half_precision and x.dtype != torch.float16:
            x = x.half()
            
        # Base network shared by all stages
        out1 = self.model0(x)
        
        # Stage 1
        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        
        # Use a single cat operation and reuse the tensor
        out2 = torch.cat([out1_1, out1_2, out1], 1)
        
        # Stage 2
        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)
        
        # Stage 3
        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)
        
        # Stage 4
        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)
        
        # Stage 5
        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)
        
        # Final stage (6)
        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)
        
        # Return outputs
        return out6_1, out6_2
    
    def forward_partial(self, stage_limit=3):
        """A faster partial forward pass that only computes up to a certain stage"""
        def partial_forward(x):
            # Base network shared by all stages
            out1 = self.model0(x)
            
            # Stage 1
            out1_1 = self.model1_1(out1)
            out1_2 = self.model1_2(out1)
            
            if stage_limit == 1:
                return out1_1, out1_2
                
            # Use a single cat operation and reuse the tensor
            out2 = torch.cat([out1_1, out1_2, out1], 1)
            
            # Stage 2
            out2_1 = self.model2_1(out2)
            out2_2 = self.model2_2(out2)
            
            if stage_limit == 2:
                return out2_1, out2_2
                
            out3 = torch.cat([out2_1, out2_2, out1], 1)
            
            # Stage 3
            out3_1 = self.model3_1(out3)
            out3_2 = self.model3_2(out3)
            
            if stage_limit == 3:
                return out3_1, out3_2
                
            out4 = torch.cat([out3_1, out3_2, out1], 1)
            
            # Stage 4
            out4_1 = self.model4_1(out4)
            out4_2 = self.model4_2(out4)
            
            if stage_limit == 4:
                return out4_1, out4_2
                
            out5 = torch.cat([out4_1, out4_2, out1], 1)
            
            # Stage 5
            out5_1 = self.model5_1(out5)
            out5_2 = self.model5_2(out5)
            
            if stage_limit == 5:
                return out5_1, out5_2
                
            out6 = torch.cat([out5_1, out5_2, out1], 1)
            
            # Final stage (6)
            out6_1 = self.model6_1(out6)
            out6_2 = self.model6_2(out6)
            
            return out6_1, out6_2
            
        return partial_forward
    
    def optimize_for_inference(self):
        """Performs optimization steps for faster inference"""
        # Try to use TorchScript JIT compilation for faster execution
        try:
            self.model0 = torch.jit.script(self.model0)
            self.model1_1 = torch.jit.script(self.model1_1)
            self.model1_2 = torch.jit.script(self.model1_2)
            self.model2_1 = torch.jit.script(self.model2_1)
            self.model2_2 = torch.jit.script(self.model2_2)
            self.model3_1 = torch.jit.script(self.model3_1)
            self.model3_2 = torch.jit.script(self.model3_2)
            self.model4_1 = torch.jit.script(self.model4_1)
            self.model4_2 = torch.jit.script(self.model4_2)
            self.model5_1 = torch.jit.script(self.model5_1)
            self.model5_2 = torch.jit.script(self.model5_2)
            self.model6_1 = torch.jit.script(self.model6_1)
            self.model6_2 = torch.jit.script(self.model6_2)
            print("Model optimized with TorchScript")
        except Exception as e:
            print(f"Could not optimize with TorchScript: {e}")
            
        return self