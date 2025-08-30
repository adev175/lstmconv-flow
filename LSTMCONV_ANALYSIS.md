# LSTMConv Model Architecture Analysis

## Kiểm tra Residual Connections trong LSTMConv

**KẾT QUẢ: ✅ CONFIRMED** - Model LSTMConv đã được tích hợp **nhiều loại Residual connections**:

### 1. ResNetBlock (Classic Residual Block)
```python
# Trong models/model_Griffin/coffin.py
class ResNetBlock(nn.Module):
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # ← RESIDUAL CONNECTION
        out = self.relu(out)
        return out
```

### 2. DecodingBlock Skip Connections
```python
# Trong models/model_Griffin/blocks_griffin.py
class DecodingBlock(nn.Module):
    def forward(self, x, res):
        x = self.up_sample(x)
        x = self.conv2d_1(x) + res  # ← SKIP CONNECTION
        x = F.relu(self.batch_norm_1(x))
        return x
```

### 3. AttentionBlock Residual Connections
```python
# Trong models/model_Griffin/blocks_griffin.py
class AttentionBlock(nn.Module):
    def forward(self, x):
        F_in = x
        x = self.conv_layers(x)
        x = self.ca_block(x)
        x = self.sa_block(x)
        return x + F_in  # ← RESIDUAL CONNECTION
```

### 4. UNet Skip Connections
```python
# Trong models/model_Griffin/Unet.py
class UNet(nn.Module):
    def forward(self, x):
        # Encoder với skip connections
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        # ...
        # Decoder với skip connections
        x = self.up1(x, s5)  # ← SKIP CONNECTION
        x = self.up2(x, s4)  # ← SKIP CONNECTION
        x = self.up3(x, s3)  # ← SKIP CONNECTION
        return x
```

---

## Mermaid Diagram: LSTMConv Model Architecture

```mermaid
graph TB
    subgraph "Input Layer"
        I0[Frame I0<br/>3 channels]
        I1[Frame I1<br/>3 channels]
        INPUT[Combined Input<br/>6 channels]
    end
    
    I0 --> INPUT
    I1 --> INPUT
    
    subgraph "Encoder Path (with Residual Connections)"
        INPUT --> E1[EncodingBlock 1<br/>6→64 channels]
        E1 --> E2[EncodingBlock 2<br/>64→128 channels]
        E2 --> E3[EncodingBlock 3<br/>128→256 channels]
        
        subgraph "EncodingBlock Detail"
            HYBRID[HybridBlock<br/>GatedConvLSTM + LocalAttention]
            BN1[BatchNorm2d]
            CONV[Conv2d]
            POOL[MaxPool2d]
            RES1[Skip Connection]
            
            HYBRID --> BN1
            BN1 --> CONV
            CONV --> RES1
            CONV --> POOL
        end
    end
    
    subgraph "Decoder Path (with Skip Connections)"
        D1[DecodingBlock 1<br/>256→128 channels]
        D2[DecodingBlock 2<br/>128→64 channels] 
        D3[DecodingBlock 3<br/>64→32 channels]
        
        E3 --> D1
        D1 --> D2
        D2 --> D3
        
        subgraph "DecodingBlock Detail"
            UP[Upsample]
            CONV1[Conv2d]
            RES2[Skip Connection ⊕]
            BN2[BatchNorm2d]
            CONV2[Conv2d]
            BN3[BatchNorm2d]
            
            UP --> CONV1
            CONV1 --> RES2
            RES2 --> BN2
            BN2 --> CONV2
            CONV2 --> BN3
        end
    end
    
    subgraph "Core Components"
        subgraph "GatedConvLSTM"
            FORGET[Forget Gate<br/>σ(Wf * [x,h])]
            INPUT_GATE[Input Gate<br/>σ(Wi * [x,h])]
            OUTPUT_GATE[Output Gate<br/>σ(Wo * [x,h])]
            CELL[Cell Gate<br/>tanh(Wc * [x,h])]
            
            CELL_STATE[Cell State<br/>ft * ct-1 + it * gt]
            HIDDEN[Hidden State<br/>ot * tanh(ct)]
        end
        
        subgraph "LocalAttention"
            ATT_CONV[Conv2d Attention]
            SIGMOID[Sigmoid]
            MUL[Element-wise Multiply]
            
            ATT_CONV --> SIGMOID
            SIGMOID --> MUL
        end
        
        subgraph "ResNetBlock"
            IDENTITY[Identity x]
            CONV_1[Conv2d]
            BN_1[BatchNorm2d]
            RELU_1[ReLU]
            CONV_2[Conv2d]
            BN_2[BatchNorm2d]
            ADD[Add ⊕]
            RELU_2[ReLU]
            
            IDENTITY --> ADD
            CONV_1 --> BN_1 --> RELU_1 --> CONV_2 --> BN_2 --> ADD --> RELU_2
        end
    end
    
    subgraph "Output"
        FLOW[Flow Prediction<br/>Conv2d 32→4]
        OUTPUT_FLOW[Optical Flow<br/>F_0_1, F_1_0<br/>4 channels]
    end
    
    D3 --> FLOW
    FLOW --> OUTPUT_FLOW
    
    %% Skip Connections
    E1 -.->|res1| D3
    E2 -.->|res2| D2  
    E3 -.->|res3| D1
    
    style RES1 fill:#ff9999
    style RES2 fill:#ff9999
    style ADD fill:#ff9999
    style INPUT fill:#e1f5fe
    style OUTPUT_FLOW fill:#e8f5e8
```

---

## Mermaid Diagram: Data Flow Pipeline

```mermaid
graph LR
    subgraph "Frame Interpolation Pipeline"
        subgraph "Input Preprocessing"
            F0[Frame I0<br/>t=0]
            F1[Frame I1<br/>t=1]
            CONCAT[Concatenate<br/>Channels]
        end
        
        F0 --> CONCAT
        F1 --> CONCAT
        
        subgraph "Edge Augmentation (EA-Net)"
            SOBEL[Sobel Edge<br/>Extractor]
            EDGE_AUG[Edge Augmentation<br/>α * I + (1-α) * I * E]
            
            CONCAT --> SOBEL
            SOBEL --> EDGE_AUG
        end
        
        subgraph "LSTMConv Flow Network"
            LSTM_NET[FlowPredictionNet<br/>Encoder-Decoder + LSTM]
            FLOW_OUT[Optical Flow<br/>F_0_1, F_1_0]
        end
        
        EDGE_AUG --> LSTM_NET
        LSTM_NET --> FLOW_OUT
        
        subgraph "Frame Synthesis"
            FLOW_COEFF[Flow Coefficients<br/>C00, C01, C10, C11]
            INTER_FLOW[Intermediate Flow<br/>F_t_0, F_t_1]
            BACKWARP[Backward Warping]
            WARP_COEFF[Warp Coefficients<br/>C0, C1]
            SYNTH[Frame Synthesis<br/>I_t]
        end
        
        FLOW_OUT --> FLOW_COEFF
        FLOW_COEFF --> INTER_FLOW
        INTER_FLOW --> BACKWARP
        F0 --> BACKWARP
        F1 --> BACKWARP
        BACKWARP --> WARP_COEFF
        WARP_COEFF --> SYNTH
        
        subgraph "Refinement (UNet)"
            UNET[UNet Refinement<br/>Skip Connections]
            FINAL[Final Interpolated<br/>Frame I_t]
        end
        
        SYNTH --> UNET
        UNET --> FINAL
    end
    
    subgraph "Training Flow"
        GT[Ground Truth<br/>Frame I_t]
        LOSS[Loss Function<br/>L1 + Perceptual]
        OPTIM[Optimizer<br/>Adam]
        
        FINAL --> LOSS
        GT --> LOSS
        LOSS --> OPTIM
    end
    
    style F0 fill:#e3f2fd
    style F1 fill:#e3f2fd
    style FINAL fill:#e8f5e8
    style LSTM_NET fill:#fff3e0
    style UNET fill:#fff3e0
    style LOSS fill:#ffebee
```

---

## Mermaid Diagram: Detailed Architecture Components

```mermaid
graph TD
    subgraph "HybridBlock Architecture"
        HB_IN[Input<br/>x, prev_state]
        GATED_LSTM[GatedConvLSTM<br/>4 Gates: f,i,o,g]
        LOCAL_ATT[LocalAttention<br/>Spatial Attention]
        HB_OUT[Output<br/>x, new_state]
        
        HB_IN --> GATED_LSTM
        GATED_LSTM --> LOCAL_ATT
        LOCAL_ATT --> HB_OUT
    end
    
    subgraph "GatedConvLSTM Gates"
        INPUT_X[Input x]
        PREV_H[Previous h]
        COMBINED[Concatenate<br/>[x, h_prev]]
        
        F_GATE[Forget Gate<br/>σ(Wf * combined)]
        I_GATE[Input Gate<br/>σ(Wi * combined)]
        O_GATE[Output Gate<br/>σ(Wo * combined)]
        G_GATE[Cell Gate<br/>tanh(Wg * combined)]
        
        C_PREV[Previous Cell<br/>c_prev]
        C_CURR[Current Cell<br/>f * c_prev + i * g]
        H_CURR[Current Hidden<br/>o * tanh(c_curr)]
        
        INPUT_X --> COMBINED
        PREV_H --> COMBINED
        COMBINED --> F_GATE
        COMBINED --> I_GATE
        COMBINED --> O_GATE
        COMBINED --> G_GATE
        
        F_GATE --> C_CURR
        I_GATE --> C_CURR
        G_GATE --> C_CURR
        C_PREV --> C_CURR
        
        O_GATE --> H_CURR
        C_CURR --> H_CURR
    end
    
    subgraph "Attention Mechanisms"
        subgraph "Channel Attention"
            CA_IN[Input Features]
            GAP[Global Average Pool]
            CA_FC1[FC: C → C/r]
            CA_RELU[ReLU]
            CA_FC2[FC: C/r → C]
            CA_SIG[Sigmoid]
            CA_MUL[Multiply ⊗]
            
            CA_IN --> GAP
            GAP --> CA_FC1
            CA_FC1 --> CA_RELU
            CA_RELU --> CA_FC2
            CA_FC2 --> CA_SIG
            CA_SIG --> CA_MUL
            CA_IN --> CA_MUL
        end
        
        subgraph "Spatial Attention"
            SA_IN[Input Features]
            SA_AVG[Average Pool<br/>Channel-wise]
            SA_MAX[Max Pool<br/>Channel-wise]
            SA_CAT[Concatenate]
            SA_CONV[Conv2d: 2→1]
            SA_SIG[Sigmoid]
            SA_MUL[Multiply ⊗]
            
            SA_IN --> SA_AVG
            SA_IN --> SA_MAX
            SA_AVG --> SA_CAT
            SA_MAX --> SA_CAT
            SA_CAT --> SA_CONV
            SA_CONV --> SA_SIG
            SA_SIG --> SA_MUL
            SA_IN --> SA_MUL
        end
    end
    
    style GATED_LSTM fill:#fff3e0
    style LOCAL_ATT fill:#e8f5e8
    style CA_MUL fill:#ff9999
    style SA_MUL fill:#ff9999
    style C_CURR fill:#e1f5fe
    style H_CURR fill:#e1f5fe
```

---

## Tóm tắt Model Architecture

### 1. **Core Components:**
- **GatedConvLSTM**: LSTM cell với 4 convolutional gates (forget, input, output, cell)
- **LocalAttention**: Spatial attention mechanism  
- **HybridBlock**: Kết hợp GatedConvLSTM + LocalAttention
- **ResNetBlock**: Classic residual block với identity skip connection

### 2. **Encoder-Decoder Architecture:**
- **EncodingBlock**: HybridBlock + BatchNorm + Conv2d + MaxPool
- **DecodingBlock**: Upsample + Conv2d với skip connections
- **U-Net Style**: Skip connections giữa encoder và decoder

### 3. **Attention Mechanisms:**
- **Channel Attention**: Squeeze-and-Excitation style với reduction ratio r=3
- **Spatial Attention**: Average + Max pooling với Conv2d
- **AttentionBlock**: Kết hợp cả hai với residual connection

### 4. **Flow Prediction:**
- **FlowPredictionNet**: Main network cho optical flow prediction  
- **EAFlowPredictionNet**: Enhanced version với edge augmentation
- **backWarp**: Backward warping cho frame interpolation

### 5. **Residual Connections:**
✅ **4 loại residual connections được implement:**
1. **ResNetBlock**: `out += identity`
2. **DecodingBlock**: `x = conv(x) + res` 
3. **AttentionBlock**: `return x + F_in`
4. **UNet**: Skip connections giữa encoder-decoder layers

### 6. **Training Strategy:**
- Loss function: L1 + Perceptual loss
- Edge augmentation cho better feature learning
- Multi-scale training với different resolutions

---

## Mermaid Diagram: Complete Training Pipeline

```mermaid
graph TB
    subgraph "Dataset Loading"
        VIMEO[Vimeo-90K Dataset<br/>Septuplet Frames]
        LOADER[DataLoader<br/>Batch Processing]
        AUG[Data Augmentation<br/>Random Crop, Flip]
    end
    
    VIMEO --> LOADER
    LOADER --> AUG
    
    subgraph "Model Selection"
        GRIFFIN[Griffin Model<br/>FlowPredictionNet]
        EA_NET[EA-Net<br/>EAFlowPredictionNet]
        UNET_MODEL[UNet Baseline]
        
        MODEL_CHOICE{Model Choice}
        MODEL_CHOICE --> GRIFFIN
        MODEL_CHOICE --> EA_NET
        MODEL_CHOICE --> UNET_MODEL
    end
    
    subgraph "Forward Pass"
        subgraph "Edge Processing (EA-Net only)"
            SOBEL_EDGE[Sobel Edge Detection]
            EDGE_WEIGHT[Learnable Edge Weight α]
            EDGE_AUG[Edge Augmentation<br/>(1-α)*I + α*(I*E)]
        end
        
        subgraph "Flow Prediction"
            ENCODER[Encoder Path<br/>3 EncodingBlocks]
            BOTTLENECK[Bottleneck<br/>256 channels]
            DECODER[Decoder Path<br/>3 DecodingBlocks]
            FLOW_HEAD[Flow Head<br/>4 channels output]
        end
        
        subgraph "Frame Synthesis"
            FLOW_SPLIT[Split Flow<br/>F_0_1, F_1_0]
            BACKWARD_WARP[Backward Warping]
            VISIBILITY[Visibility Maps]
            BLEND[Alpha Blending]
        end
    end
    
    AUG --> SOBEL_EDGE
    SOBEL_EDGE --> EDGE_WEIGHT
    EDGE_WEIGHT --> EDGE_AUG
    EDGE_AUG --> ENCODER
    
    ENCODER --> BOTTLENECK
    BOTTLENECK --> DECODER
    DECODER --> FLOW_HEAD
    FLOW_HEAD --> FLOW_SPLIT
    FLOW_SPLIT --> BACKWARD_WARP
    BACKWARD_WARP --> VISIBILITY
    VISIBILITY --> BLEND
    
    subgraph "Loss Computation"
        GT_FRAME[Ground Truth Frame]
        PRED_FRAME[Predicted Frame]
        
        subgraph "Loss Functions"
            L1_LOSS[L1 Loss<br/>|pred - gt|]
            PERCEPTUAL[Perceptual Loss<br/>VGG Features]
            FLOW_LOSS[Flow Consistency Loss]
            TOTAL_LOSS[Total Loss<br/>L1 + λ₁*Perceptual + λ₂*Flow]
        end
    end
    
    BLEND --> PRED_FRAME
    PRED_FRAME --> L1_LOSS
    GT_FRAME --> L1_LOSS
    PRED_FRAME --> PERCEPTUAL
    GT_FRAME --> PERCEPTUAL
    FLOW_SPLIT --> FLOW_LOSS
    
    L1_LOSS --> TOTAL_LOSS
    PERCEPTUAL --> TOTAL_LOSS
    FLOW_LOSS --> TOTAL_LOSS
    
    subgraph "Optimization"
        ADAM[Adam Optimizer<br/>lr=1e-4]
        SCHEDULER[LR Scheduler<br/>StepLR]
        BACKPROP[Backpropagation]
        UPDATE[Parameter Update]
    end
    
    TOTAL_LOSS --> BACKPROP
    BACKPROP --> ADAM
    ADAM --> UPDATE
    UPDATE --> SCHEDULER
    
    subgraph "Evaluation Metrics"
        PSNR[Peak Signal-to-Noise Ratio]
        SSIM[Structural Similarity Index]
        LPIPS[Learned Perceptual Image Patch Similarity]
    end
    
    PRED_FRAME --> PSNR
    PRED_FRAME --> SSIM
    PRED_FRAME --> LPIPS
    GT_FRAME --> PSNR
    GT_FRAME --> SSIM
    GT_FRAME --> LPIPS
    
    style GRIFFIN fill:#fff3e0
    style EA_NET fill:#e8f5e8
    style TOTAL_LOSS fill:#ffebee
    style UPDATE fill:#e3f2fd
```

---

## Implementation Details và Code Structure

### File Organization:
```
models/
├── model_Griffin/          # Main LSTMConv implementation
│   ├── coffin.py          # GatedConvLSTM, LocalAttention, ResNetBlock
│   ├── hybrid.py          # HybridBlock (LSTM + Attention)
│   ├── blocks_griffin.py  # EncodingBlock, DecodingBlock, AttentionBlocks
│   ├── flowpred2_griffin.py # FlowPredictionNet
│   ├── synthesis2.py      # Frame synthesis utilities
│   └── Unet.py           # UNet with skip connections
└── model_4/               # Enhanced EA-Net implementation
    ├── ea_flow_net.py     # EAFlowPredictionNet with edge augmentation
    ├── blocks_griffin.py  # Similar blocks with edge support
    └── training_strategy.py # Training configurations
```

### Key Innovations:
1. **Temporal Consistency**: GatedConvLSTM maintains temporal information across frames
2. **Multi-Scale Features**: Encoder-decoder với skip connections ở multiple scales  
3. **Attention Mechanisms**: Channel + Spatial attention cho feature refinement
4. **Edge-Aware Processing**: Sobel edge detection + learnable augmentation
5. **Residual Learning**: Multiple residual connections cho gradient flow