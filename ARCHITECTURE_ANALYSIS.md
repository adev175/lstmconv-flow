# LSTMConv Model Architecture Analysis

## Kiểm tra Residual Connections

### ✅ **Kết quả: Model đã có bộ Residual**

Model LSTMConv trong repository này **ĐÃ ĐƯỢC TÍCH HỢP** các loại Residual connections khác nhau:

#### 1. **ResNetBlock - Classic Residual Connection**
```python
class ResNetBlock(nn.Module):
    def forward(self, x):
        identity = x
        # ... convolution layers ...
        out += identity  # ← RESIDUAL CONNECTION
        return out
```

#### 2. **DecodingBlock - U-Net Style Skip Connections**
```python
def forward(self, x, res):
    x = self.up_sample(x)
    x = self.conv2d_1(x) + res  # ← SKIP CONNECTION (residual từ encoder)
    return x
```

#### 3. **AttentionBlock - Residual Attention**
```python
def forward(self, x):
    F_in = x
    x = self.conv_layers(x)
    x = self.ca_block(x)
    x = self.sa_block(x)
    return x + F_in  # ← RESIDUAL CONNECTION
```

## Tóm tắt Kiến trúc Model

### Core Components:

1. **GatedConvLSTM**: LSTM với convolutional gates
2. **LocalAttention**: Attention mechanism cục bộ
3. **HybridBlock**: Kết hợp GatedConvLSTM + LocalAttention
4. **EncodingBlock**: Encoder với LSTM temporal processing
5. **DecodingBlock**: Decoder với skip connections
6. **FlowPredictionNet**: Mạng chính dự đoán optical flow

### Residual Connections Summary:

| Component | Residual Type | Implementation |
|-----------|---------------|----------------|
| ResNetBlock | Classic Residual | `out += identity` |
| DecodingBlock | Skip Connection | `x + res` (from encoder) |
| AttentionBlock | Residual Attention | `output + input` |
| LocalAttention | Gating (not residual) | `x * attention_weights` |

### Key Features:
- ✅ Multiple residual connection types
- ✅ Temporal processing với LSTM
- ✅ Spatial attention mechanisms
- ✅ U-Net style encoder-decoder
- ✅ Skip connections for gradient flow

## Mermaid Diagrams

Xem các file diagrams riêng biệt:
- `model_architecture.mmd`: Kiến trúc tổng thể
- `data_flow.mmd`: Luồng dữ liệu chi tiết
- `component_relationships.mmd`: Mối quan hệ giữa các components